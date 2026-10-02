# scripts/build_pakwheels_dataset.py
"""
Builds data/pakwheels/pakwheels_support.jsonl, the fine-tuning set for the
PakWheels customer-support demo.

Design:
- Each intent has customer phrasings (English, Roman Urdu, typos) and answer
  variants written from data/pakwheels/knowledge_base.md (paraphrased from
  pakwheels.com, never copied).
- Answers WITHOUT context never state prices, fees or other figures that
  change; they point to the right page. Answers WITH context (the same KB
  section the RAG index retrieves) do state them. This teaches the model to
  quote figures only when retrieval supplies them.
- A few rows pair a fee question with an unrelated KB section (a retrieval
  miss) and the no-figures answer, so the model doesn't invent numbers.
- Bank account numbers never appear; payment questions go to the official page.

v2 (after the first demo run):
- Specific answers: car model -> engine-size tier, city -> yes/no for each
  service, model year -> eligibility, sale price -> commission, number of days
  -> featured price. Eligibility answers start with Yes or No.
- Confusable sections: grounded rows often include a look-alike section (e.g.
  Sell It For Me fees next to inspection prices), and "wrong section only" rows
  answer from the intent while ignoring the retrieved text.
- Rebalanced: every fee/eligibility phrasing gets a grounded row.

v3 (after the second demo run, asked as one long chat):
- Multi-turn rows (`history`): the model copied the previous answer when the
  topic changed (Multan got the inspection-cities answer). Topic-switch rows put
  an earlier, often look-alike, exchange in the history; follow-up rows ("And
  for my Prado?") answer the same service for the new car or city.
- "I don't like the colour" (change of mind: no) vs "received a different
  colour" (wrong item: yes).
- The demo questions in scripts/demo_eval.py never appear in training rows.

v4 (after the third demo run, 25/30):
- Many more engine sizes and sale prices (half of them at or under the 5 lakh
  commission minimum), with the deciding comparison stated first.
- Intents without a KB section (scams, tampering, personal data, orders) get
  rows with the irrelevant section retrieval actually returns, answered as if
  there were no context. Payment questions get a grounded row per phrasing.
- More phrasings for advance-payment scams, bank-account requests and odometer
  or document tampering.

v5 (after the fourth demo run, 29/30):
- "Gari bechni hai, ad kaise lagaun?" got the featured-ad payment steps: retrieval
  returns both sections, and payment rows outnumbered post-ad rows. Post-ad gets a
  grounded row per phrasing and more Roman Urdu phrasings; post-ad and payment
  rows mostly carry each other's section (`hard_distractor`), so the question,
  not the section, picks the answer.

v6 (after the first --conversation run, 28/30):
- "Sell It For Me Lahore mein available hai?" got the bare city list: the old
  eligibility intents mixed city-specific questions with a list-only answer.
  They now keep general questions; every specific city gets Yes/No.
- "Can PakWheels transfer my car in Islamabad?" was answered as Sell It For Me
  (same sentence shape). Transfer cities get their own Yes/No rows, and the two
  sections are look-alikes for each other.

v7 (after the fresh 54-question test, 43 pass): verified answers.
- Prices for a car, engine size, duration or sale price, and Yes/No for a city
  or model year, are looked up by src/facts.py and put at the top of the
  context as "## Verified answer", in training exactly as in the app. Any row
  whose answer disagrees with the verified answer is replaced by it, so the
  model learns to restate it. The app also blocks figures not in the context.
- Many more off-topic requests (translation, writing, coding, general
  knowledge) and Roman Urdu city phrasings.

Run from the repo root:  python scripts/build_pakwheels_dataset.py
"""

import json
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.kb import sections_by_title  # noqa: E402
from src.facts import (  # noqa: E402  (the answer wording the app's verified answers use)
    AUCTION_FEE_ANSWER, bundle_answer, INSPECTION_CITIES, LARGE_INSPECTION_ANSWER, LARGE_SIFM_ANSWER,
    OPEN_LETTER_ANSWER, SIFM_CITIES, TRANSFER_CITIES, car_phrase, cc_phrase, city_list,
    commission_answer, commission_for, detect, featured_answer, FEATURED, inspection_answer,
    inspection_city_answer, lakh_label, model_year_answer, sifm_city_answer, sifm_fee_answer,
    strip_fact_block, transfer_city_answer, FACT_HEADER,
)

KB_PATH = "data/pakwheels/knowledge_base.md"
OUT_PATH = "data/pakwheels/pakwheels_support.jsonl"
SEED = 42
CONTEXT_EXAMPLES_PER_INTENT = 4      # intents without figures
WRONG_SECTION_EXAMPLES_PER_INTENT = 2
RETRIEVAL_MISS_EXAMPLES = 15
CONFUSABLE_DISTRACTOR_RATE = 0.6
HARD_DISTRACTOR_RATE = 0.8  # intents with `hard_distractor`: share of distractors that are it
IRRELEVANT_CONTEXT_EXAMPLES_PER_INTENT = 2  # intents without a KB section
TOPIC_SWITCH_EXAMPLES = 120
TOPIC_SWITCH_NO_CONTEXT_RATE = 0.35  # the new question has no KB section (safety, scope, ...)
TWO_PREVIOUS_TURNS_RATE = 0.3        # the app keeps up to 2 previous exchanges

# Sections that retrieval mixes up; used as hard distractors
CONFUSABLE = {
    "Sell It For Me fees": ["Car inspection prices", "Auction sheet verification", "Featured ad prices"],
    "Car inspection prices": ["Sell It For Me fees", "Featured ad prices", "Car inspection coverage and process"],
    "Car inspection coverage and process": ["Car inspection prices", "Sell It For Me eligibility and process"],
    "Car inspection cities": ["Sell It For Me eligibility and process", "Car registration and ownership transfer service"],
    "Sell It For Me eligibility and process": ["Car inspection cities", "Sell It For Me fees",
                                               "Car registration and ownership transfer service"],
    "Featured ad prices": ["How to feature an ad and pay", "Sell It For Me fees"],
    "How to feature an ad and pay": ["Featured ad prices", "Posting, closing and reopening ads"],
    "Posting, closing and reopening ads": ["How to feature an ad and pay", "Featured ad prices"],
    "Auction sheet verification": ["Sell It For Me fees", "Car inspection prices"],
    "Car insurance": ["Car finance"],
    "Car finance": ["Car insurance"],
    "AutoStore returns and refunds": ["AutoStore payment methods"],
    "AutoStore payment methods": ["AutoStore returns and refunds"],
    "Contact and support hours": ["How to feature an ad and pay", "Auction sheet verification"],
    "Account, login and privacy": ["Posting, closing and reopening ads"],
    "Inspected vs Certified cars": ["Car inspection coverage and process"],
    "Car registration and ownership transfer service": ["Car inspection cities",
                                                        "Sell It For Me eligibility and process"],
    "Tools for buyers": ["Other PakWheels services and community"],
    "Other PakWheels services and community": ["Tools for buyers"],
}

HELPLINE = "042-111-943-357 (042-111-WHEELS)"
HOURS = "Monday to Sunday, 9 am to 9 pm"
HUMAN = (f"the PakWheels helpline at {HELPLINE}, {HOURS}, or email info@pakwheels.com")

# kb: KB section title(s) that answer the intent; used for context rows.
# context_responses: answers that may quote figures from the context.
INTENTS = [
    # ---------------------------------------------------------------- general
    dict(intent="greeting", category="GENERAL", questions=[
        "hi", "hello", "Assalam o Alaikum", "salam", "hey there", "good evening",
        "AOA, kya aap meri madad kar sakte hain?", "hello is anyone there?",
    ], responses=[
        "Hello! I'm the PakWheels virtual assistant. I can help with buying or selling a car, "
        "your ads and account, and services like inspection, Sell It For Me, auction sheet "
        "verification, finance and insurance. What can I help you with?",
        "Wa Alaikum Assalam! How can I help you today? You can ask me about posting an ad, "
        "buying a used car, PakWheels services or your account.",
    ]),
    dict(intent="thanks_goodbye", category="GENERAL", questions=[
        "thanks", "thank you so much", "shukriya", "ok bye", "that's all, thanks",
        "great, that helped", "JazakAllah", "thanks bro",
    ], responses=[
        "You're welcome! If you have any other car or PakWheels question, I'm here 24/7.",
        "Glad I could help. Good luck with your car, and feel free to come back anytime.",
    ]),
    dict(intent="bot_capabilities", category="GENERAL", questions=[
        "what can you do?", "are you a human?", "who are you", "are you a bot",
        "what kind of questions can I ask you?", "aap kaun ho?", "can you help me sell my car",
        "is this pakwheels customer support?",
    ], responses=[
        "I'm the PakWheels virtual assistant, an AI that answers questions any time of day. "
        "I can explain how to post and manage ads, how to buy safely, and PakWheels services "
        "such as car inspection, Sell It For Me, featured ads, auction sheet verification, "
        "registration and transfer, finance, insurance and AutoStore orders. I can't see your "
        f"account or orders; for those, contact {HUMAN}.",
    ]),
    dict(intent="human_agent", category="CONTACT", kb=["Contact and support hours"], questions=[
        "I want to talk to a real person", "connect me to an agent", "customer care number?",
        "how do I contact pakwheels", "helpline number kya hai?", "what is your phone number",
        "can I email you?", "I need to speak with someone from pakwheels",
        "where is pakwheels office", "pakwheels head office address",
    ], responses=[
        f"You can reach PakWheels support at {HELPLINE} or info@pakwheels.com. Human agents are "
        f"available {HOURS}. You can also send a message through the Contact Us form on the "
        "website at any time. I'm happy to help in the meantime if you tell me what you need.",
    ], context_responses=[
        f"PakWheels support: call {HELPLINE} or email info@pakwheels.com, {HOURS}. The head "
        "office is at Saeed Alam Tower, 37 Commercial Zone, Liberty Market, Gulberg, Lahore. "
        "Outside those hours you can use the Contact Us form, and I can answer general "
        "questions right now.",
    ]),
    dict(intent="support_hours", category="CONTACT", kb=["Contact and support hours"], questions=[
        "what are your timings?", "is support available 24/7?", "can I call at midnight?",
        "office hours?", "support kab tak available hai?", "are you open on sunday",
        "when can I call customer service",
    ], responses=[
        f"Human support is available {HOURS} on {HELPLINE}. I'm the virtual assistant and I'm "
        "available 24/7 for general questions; anything that needs your account can be sent "
        "through the Contact Us form and the team will get back to you.",
    ]),
    dict(intent="complaint", category="CONTACT", questions=[
        "I want to file a complaint", "very bad experience with your service",
        "your inspector was late and rude", "I am not happy with pakwheels",
        "mujhe complaint karni hai", "who do I escalate an issue to?",
        "nobody is responding to my request",
    ], responses=[
        "I'm sorry about your experience. I can't open a case myself, but the support team can: "
        f"call {HELPLINE} ({HOURS}) or email info@pakwheels.com with your name, phone number, "
        "the service involved (for example inspection, ad or order) and what happened. If you "
        "share the details here, I can help you write the message.",
    ]),
    # ---------------------------------------------------------------- account
    dict(intent="create_account", category="ACCOUNT", kb=["Account, login and privacy"], questions=[
        "how do I sign up?", "how to create a pakwheels account", "register kaise karun?",
        "can I login with google?", "sign in with phone number", "do I need an account to post an ad",
        "can i use facebook to login",
    ], responses=[
        "Click Sign Up or Sign In at the top of the site or app. You can continue with your "
        "mobile number (you'll get an OTP), Google, Facebook or email. You need to be signed in "
        "to post an ad.",
    ]),
    dict(intent="activation_email", category="ACCOUNT", kb=["Account, login and privacy"], questions=[
        "I didn't get the activation email", "account confirmation email not received",
        "verification email nahi aayi", "activation link missing", "never got the email to verify my account",
    ], responses=[
        "First check your spam or junk folder. If it isn't there, try signing in with your "
        "email and password; the site detects the account isn't active and offers a link to "
        "resend the activation email. You can also sign in with your mobile number or Google "
        f"instead. If it still doesn't work, contact {HUMAN}.",
    ]),
    dict(intent="forgot_password", category="ACCOUNT", kb=["Account, login and privacy"], questions=[
        "forgot my password, please help", "reset password", "password bhool gaya hoon",
        "can't log in, forgot password", "how to change my password", "i lost my password help",
        "password reset link not working",
    ], responses=[
        "On the Sign In screen choose Forgot Password, enter the email on your account and open "
        "the reset link sent to you, then set a new password. Check spam if the email doesn't "
        "arrive. If you normally sign in with your mobile number, Google or Facebook, you can use "
        "that instead. Never share your password or OTP with anyone.",
    ]),
    dict(intent="otp_issue", category="ACCOUNT", questions=[
        "I'm not receiving the OTP", "OTP nahi aa raha", "code not coming on my phone",
        "it says too many OTP attempts", "otp expired", "verification code sms not received",
    ], responses=[
        "Wait for the timer on the screen to finish, then request a new code. The login screen "
        "lets you resend the code by SMS, WhatsApp or call. If you see a 'too many attempts' "
        "message, wait a while before trying again, and check that your number is entered "
        "correctly. You can also sign in with Google, Facebook or email. Never share your OTP "
        "with anyone, including people who say they are from PakWheels.",
    ]),
    dict(intent="edit_profile", category="ACCOUNT", questions=[
        "how do I edit my profile", "change my name on pakwheels", "update profile picture",
        "profile edit kaise karun", "how to change my avatar",
    ], responses=[
        "Sign in, open your profile (My PakWheels) and choose Edit Profile. There you can update "
        "your details and upload a profile picture in JPEG, PNG or GIF format.",
    ]),
    dict(intent="delete_account", category="ACCOUNT", questions=[
        "how do I delete my account", "I want to remove my pakwheels account permanently",
        "account delete karna hai", "close my account and delete my data",
    ], responses=[
        "I don't have a self-service option to share for deleting an account. Please send the "
        "request through the Contact Us form (choose the Account subject) or email "
        "info@pakwheels.com from the email on your account, and the team will guide you. If you "
        "only want to stop selling, you can close your ads from My Ads instead.",
    ]),
    # ---------------------------------------------------------------- selling
    # Retrieval returns the payment section next to this one, and the model once
    # answered "ad kaise lagaun?" with the featured-ad payment steps
    dict(intent="post_ad_car", category="SELLING", kb=["Posting, closing and reopening ads"],
         ground_all=True, hard_distractor="How to feature an ad and pay", questions=[
        "how do I post an ad for my car?", "I want to sell my car", "gari bechni hai, ad kaise post karun?",
        "is posting an ad free?", "how to list my car on pakwheels", "sell my corolla online",
        "steps to post ad", "can i post an ad from the app", "post ad charges?",
        "car sell karne ka tareeqa batao",
        "apni car ka ad free mein laga sakta hoon?", "mujhe apni gari sell karni hai",
        "do I have to pay to post an ad?", "how to put my car up for sale",
        "car ka ad kaise dalun?", "gaari ka ad lagana hai, kya karna hoga?",
        "apni car bechne ke liye ad kaise lagayein?", "ad lagane ke kitne paise lagte hain?",
        "used car bechne ka ad kahan lagaun?", "I want to put up an ad to sell my car",
        "how do I advertise my car on pakwheels?", "meri car ka ad post kar do",
    ], responses=[
        "Posting a car ad is free. Sign in, choose Post an Ad (or Sell Your Car under Used Cars), "
        "fill in the car's details, price and photos, and submit. Ads are reviewed before going "
        "live and you'll be notified once approved. Clear photos and complete details help you "
        "get more calls. If you'd rather not handle buyers yourself, PakWheels Sell It For Me "
        "can sell it for you.",
    ]),
    dict(intent="post_ad_bike_parts", category="SELLING", kb=["Posting, closing and reopening ads"], questions=[
        "how do I sell my bike?", "can I sell car parts on pakwheels", "post ad for motorcycle",
        "bike bechni hai", "sell my alloy rims", "list accessories for sale",
    ], responses=[
        "Yes. Use Post an Ad and choose Sell Your Bike or Sell Accessory (for parts and "
        "accessories). Posting is free; fill in the details and photos and submit. The ad is "
        "reviewed before it appears in listings.",
    ]),
    dict(intent="ad_not_live", category="SELLING", kb=["Posting, closing and reopening ads"], questions=[
        "my ad is not showing", "why is my ad still pending?", "ad approve kab hoga?",
        "how long does ad approval take", "I posted my ad yesterday but can't find it",
        "ad live nahi hua",
    ], responses=[
        "New and reopened ads are reviewed by the PakWheels team before they appear in search, "
        "and you're notified once approved. The team may call to verify the ad, so keep your "
        "phone reachable. I can't see your account, so if your ad has been pending for a long "
        f"time, contact {HUMAN} with your Ad ID.",
    ]),
    dict(intent="ad_rejected", category="SELLING", kb=["Posting, closing and reopening ads"], questions=[
        "why did my ad get rejected?", "ad rejected email aayi hai", "my car ad got declined",
        "pakwheels removed my ad why", "reasons for ad rejection",
    ], responses=[
        "Common reasons are incomplete or incorrect car details, photos or text that break the "
        "terms of service, or the team not being able to reach you by phone to verify the ad. "
        "Fix the details and submit again. For the exact reason on your ad, contact "
        f"{HUMAN} with your Ad ID.",
    ]),
    dict(intent="close_ad", category="SELLING", kb=["Posting, closing and reopening ads"], questions=[
        "I sold my car, how do I remove the ad?", "close my ad", "ad band karna hai",
        "don't want to sell anymore", "how to delete my ad", "stop the calls, car is sold",
    ], responses=[
        "Sign in, go to My Ads, find the ad and choose Close. It will stop showing in active "
        "listings, and you can reopen it later if you change your mind.",
    ]),
    dict(intent="reopen_ad", category="SELLING", kb=["Posting, closing and reopening ads"], questions=[
        "my ad expired, how do I reopen it?", "reactivate my old ad", "ad expire ho gaya",
        "how to repost my closed ad", "bring back expired listing",
    ], responses=[
        "Sign in, open My Ads, find the expired or closed ad and choose to open it again. It "
        "goes back to the team for a quick review and appears in listings once approved.",
    ]),
    dict(intent="edit_ad", category="SELLING", questions=[
        "how do I change the price of my ad?", "edit my ad", "ad mein price change karni hai",
        "update photos on my listing", "I made a mistake in my ad details",
    ], responses=[
        "Sign in and open My Ads to manage your listing. If you can't change a detail there, "
        "you can close the ad and post a corrected one, or contact "
        f"{HUMAN} with your Ad ID. Edited ads may be reviewed again before they go live.",
    ]),
    dict(intent="seller_privacy", category="SELLING", kb=["Account, login and privacy"], questions=[
        "will my email be shown to buyers?", "is my email public?", "meri email show hogi?",
        "do buyers see my email address", "privacy of my contact details",
    ], responses=[
        "No, your email address is not shown to buyers. Messages buyers send are forwarded to "
        "you while your email stays hidden, and you can report spam or abuse. Be careful what "
        "you share in your ad description, and never share an OTP.",
    ]),
    dict(intent="featured_ad_info", category="SELLING", kb=["Featured ad prices"], questions=[
        "what is a featured ad?", "how can I sell my car faster?", "featured ad ka faida?",
        "is a featured ad worth it", "how to get more calls on my ad", "boost my ad",
    ], responses=[
        "A featured ad stays at the top of search results and on the Used Cars main page with a "
        "Featured tag, while free ads slip down as new ones are posted. PakWheels says featured "
        "ads get many times more responses. You can choose how many days to feature it on the "
        "Feature Your Ad page.",
    ]),
    dict(intent="featured_ad_price", category="SELLING", kb=["Featured ad prices"], questions=[
        "how much does a featured ad cost?", "featured ad price", "feature karne ke kitne paise?",
        "7 days featured ad charges", "price of featuring my ad for a month",
        "do you have packages for dealers?", "feature credit price",
    ], responses=[
        "Featured ad prices depend on how long you feature the ad (for example 7, 14 or 28 "
        "days), and dealers can buy bundles of feature credits at a lower cost per ad. Prices "
        "change, so check the current list on the Feature Your Ad page.",
    ], context_responses=[
        "Listed prices for featuring one ad: 7 days PKR 2,950, 14 days PKR 4,450 and 28 days "
        "PKR 5,950. Bundles of feature credits cost less per ad, for example 5 credits for PKR "
        "13,450. Prices can change, so confirm on the Feature Your Ad page before paying.",
    ]),
    dict(intent="feature_payment", category="SELLING", kb=["How to feature an ad and pay"],
         ground_all=True, hard_distractor="Posting, closing and reopening ads", questions=[
        "how do I pay for a featured ad?", "how to feature my ad", "payment method for featured ad",
        "send me your bank account number", "feature ad ki payment kaise karun", "can I pay by cheque",
        "someone asked me to pay for featuring in his account", "jazzcash se feature payment ho sakti hai?",
        "what's your account number for the featured ad payment?", "IBAN bhej do, feature ka payment karna hai",
        "where do I send money to feature my ad?", "give me the bank details to pay for featuring",
        "share your easypaisa number so I can pay the feature fee",
        "can I transfer the feature fee straight to your account?",
    ], responses=[
        "Post your ad first and note its Ad ID. Then pay using a method listed on the official "
        "How To Pay section of the Feature Your Ad page and call "
        f"{HELPLINE} with the payment details; the ad is featured once payment is confirmed. "
        "I don't share account numbers in chat. Use only the details on the official page, and "
        "never pay a personal account or someone who contacts you claiming to be PakWheels.",
    ]),
    dict(intent="sifm_info", category="SELLING", kb=["Sell It For Me eligibility and process"], questions=[
        "what is sell it for me?", "can pakwheels sell my car for me?", "sell it for me kaise kaam karta hai",
        "I don't have time to deal with buyers", "how does managed selling work",
        "who handles calls in sell it for me", "how long does sell it for me take",
    ], responses=[
        "With Sell It For Me, PakWheels sells your car for you. You sign up, the team visits to "
        "inspect and photograph the car and posts a featured ad, a dedicated sales rep handles "
        "calls, offers and negotiation, and PakWheels helps with paperwork and a secure payment "
        "once the deal is done. You can start from the Sell It For Me page.",
    ], context_responses=[
        "Sell It For Me works in five steps: you sign up; the team visits, inspects and "
        "photographs the car and posts a featured ad; the ad is promoted on the site and app; a "
        "sales rep handles calls, offers and negotiation (aiming to close within 45 days); and "
        "PakWheels arranges paperwork and secure payment once the deal is locked.",
    ]),
    dict(intent="sifm_fee", category="SELLING", kb=["Sell It For Me fees"], questions=[
        "how much does sell it for me cost?", "sell it for me charges", "commission kitna hai?",
        "is the sell it for me fee refundable?", "what if my car doesn't sell, do I get my money back",
        "fee for selling an SUV through pakwheels",
    ], responses=[
        "Sell It For Me has a non-refundable onboarding fee that depends on the "
        "car's engine size and type, plus a commission charged only after the car is sold. The "
        "current amounts are on the Sell It For Me page.",
    ], context_responses=[
        "There is a non-refundable onboarding fee at sign-up: PKR 2,000 up to 1000cc, "
        "PKR 5,000 for 1001-2000cc, and PKR 7,000 for SUVs, 4x4s, jeeps and German cars. After "
        "the sale, the commission is 1% of the selling price, with a minimum of PKR 5,000 if the "
        "car sells for 5 lakh or less.",
    ]),
    # General questions only: a specific city, year or open letter gets a Yes/No
    # answer from gen_city_eligibility / gen_model_year. Mixing them here taught
    # the model to answer "available in Lahore?" with the bare city list.
    dict(intent="sifm_eligibility", category="SELLING", kb=["Sell It For Me eligibility and process"], questions=[
        "which cities have sell it for me", "where is sell it for me available?",
        "what are the conditions for sell it for me?", "who can use sell it for me?",
        "sell it for me kin shehron mein hai?", "requirements for sell it for me",
    ], responses=[
        "Sell It For Me is offered in selected major cities and has conditions on the car's "
        "model year and documents; open-letter cars are not eligible. Check the Sell It For Me "
        "page for the current list of cities, or I can check if you tell me your city and car.",
    ], context_responses=[
        "Sell It For Me is available in Karachi, Lahore, Faisalabad, Gujranwala, Islamabad, "
        "Rawalpindi and Peshawar. The car must be model year 2000 or newer, and open-letter "
        "cars are not eligible. If your city isn't listed, you can still post a free ad yourself.",
    ]),
    dict(intent="price_my_car", category="SELLING", kb=["Tools for buyers"], questions=[
        "what price should I ask for my car?", "how much is my car worth?", "meri gari ki value kya hai",
        "estimate price of my 2015 city", "how do I price my car to sell fast",
    ], responses=[
        "Use the PakWheels Price Calculator under Used Cars: enter the make, model, year, "
        "mileage and city to get a market estimate. Comparing similar ads in your city also "
        "helps. A PakWheels inspection report can make buyers more confident and support your "
        "asking price. I can't quote a value myself because prices change daily.",
    ]),
    # ---------------------------------------------------------------- buying
    dict(intent="search_tips", category="BUYING", questions=[
        "how do I search for cars under 20 lakh?", "find cars in lahore", "filter by year and city",
        "automatic cars under 30 lakh karachi", "search by budget", "show me 660cc cars",
        "how to find old expired ads", "search cars newer than 2018",
    ], responses=[
        "Go to Used Cars and use the search filters: city, make and model, price range, year "
        "range, mileage, transmission and more. You can also browse categories such as budget "
        "ranges, body type, 660cc or automatic cars. Results show active ads by default; you "
        "can refine them further from the filter panel.",
    ]),
    dict(intent="used_car_value", category="BUYING", kb=["Tools for buyers"], questions=[
        "is this price fair for a 2017 corolla?", "what's the market price of a used civic",
        "used car price check", "kya yeh price theek hai?", "how to know if a car is overpriced",
    ], responses=[
        "Check the PakWheels Price Calculator for a market estimate and compare similar ads "
        "(same model, year, mileage and city). Condition matters a lot, so a PakWheels "
        "inspection before you pay is the best way to know whether the price is fair. I can't "
        "quote live market prices myself.",
    ]),
    dict(intent="new_car_prices", category="BUYING", kb=["Tools for buyers"], questions=[
        "what is the price of a new honda civic?", "new car price list", "corolla ki nayi price?",
        "on road price of alto", "latest price of kia sportage", "what does a new yaris cost",
    ], responses=[
        "New car prices change often, so I don't quote them. The New Car Prices page lists "
        "current prices by make and model, and the On Road Price calculator adds registration "
        "and other costs. Reviews and comparisons are also on the New Cars section.",
    ]),
    dict(intent="car_recommendation", category="BUYING", questions=[
        "which car is best under 30 lakh?", "civic or corolla?", "best family car in pakistan",
        "konsi gari leni chahiye?", "suggest a fuel efficient car", "city vs yaris which is better",
        "best car for a new driver",
    ], responses=[
        "It depends on your budget, how you'll use it (city or highway), fuel economy, "
        "maintenance and parts cost, and resale value. Use Car Comparisons on PakWheels to "
        "compare specs side by side and read owner Reviews. For used cars, shortlist a few ads "
        "and get a PakWheels inspection before you decide. Tell me your budget and needs and I "
        "can suggest what to compare.",
    ]),
    dict(intent="genuine_ads", category="BUYING", questions=[
        "are the ads genuine?", "how do I know a seller is real?", "is it safe to buy from pakwheels",
        "fake ads hote hain?", "how does pakwheels verify ads",
    ], responses=[
        "PakWheels reviews ads before they go live and verifies new sellers by phone, but it's "
        "a marketplace, so stay careful: meet in a safe public place, check the original "
        "documents and the chassis and engine numbers, verify the car on the official MTMIS "
        "service, get a PakWheels inspection, and never pay an advance before seeing the car.",
    ]),
    dict(intent="scam_report", category="BUYING", ground_all=True,
         near=["Sell It For Me fees", "How to feature an ad and pay", "Car finance",
               "Account, login and privacy"], questions=[
        "seller is asking for advance payment", "I think this ad is a scam",
        "someone called saying he is from pakwheels and asked for my OTP",
        "fraud seller, what do I do?", "buyer sent fake payment screenshot",
        "advance mang raha hai, kya karun?",
        "seller wants a token payment to hold the car", "is it safe to send a deposit to book a car?",
        "the seller wants money before he shows me the car", "car dekhne se pehle paise mang raha hai",
        "seller says pay a booking amount first, then visit", "should I pay half in advance before the viewing?",
        "the seller only shares a bank account and refuses to meet",
        "buyer asked me to pay a refundable verification fee",
    ], responses=[
        "Please don't send any money or share any OTP. PakWheels staff will never ask for your "
        "OTP or password. Stop contacting the person, keep screenshots and the ad link, and "
        f"report it to {HUMAN}. If you've already paid, also contact your bank or wallet "
        "provider immediately and report it to the authorities.",
    ]),
    dict(intent="alerts_saved_ads", category="BUYING", kb=["Tools for buyers"], questions=[
        "can I get notified when a car is listed?", "how to save an ad", "create alert for corolla",
        "notification for new ads", "save favourite cars",
    ], responses=[
        "Yes. Sign in and create a Quick Alert from your search to be notified when new matching "
        "listings appear, and use the save (heart) option on ads to keep them in your saved ads.",
    ]),
    dict(intent="certified_vs_inspected", category="BUYING", kb=["Inspected vs Certified cars"], questions=[
        "what does pakwheels certified mean?", "difference between inspected and certified",
        "certified car warranty", "are managed by pakwheels cars inspected?", "certified gari kya hoti hai",
    ], responses=[
        "A PakWheels Inspected car has an inspection report describing its condition, but "
        "PakWheels doesn't approve the car. PakWheels Certified cars are inspected on 200+ "
        "points, must be non-accidental with a good engine and gearbox, and come with a free "
        "repair warranty. You can filter for both in Used Cars search.",
    ]),
    dict(intent="dealers", category="BUYING", questions=[
        "find car dealers near me", "list of used car dealers in islamabad", "new car dealership karachi",
        "dealer kaise dhundun", "become a dealer on pakwheels",
    ], responses=[
        "Used Cars and New Cars each have a Dealers section where you can find dealers by city. "
        "If you're a dealer who wants to join the PakWheels dealership network or feature "
        "listings in bulk, send a request through the Contact Us page (Partnerships).",
    ]),
    dict(intent="buying_documents", category="BUYING", questions=[
        "what documents should I check before buying a used car?", "open letter car lena chahiye?",
        "is it safe to buy a car with an open transfer letter", "documents for buying used car",
        "how to check if car is stolen",
    ], responses=[
        "Check the original registration book or smart card, the seller's CNIC, tax/token "
        "status, previous transfer letters and receipts, and that the chassis and engine "
        "numbers match the documents. Avoid open-letter cars and verify the vehicle on the "
        "official MTMIS service for its province. For imported cars, verify the auction sheet. "
        "PakWheels inspection also checks documents and vehicle history where available.",
    ]),
    # ---------------------------------------------------------------- services
    dict(intent="inspection_info", category="SERVICES", kb=["Car inspection coverage and process"], questions=[
        "what does pakwheels inspection check?", "how long does inspection take?",
        "when will I get the inspection report", "inspection report kaise milegi",
        "do you inspect hybrid cars?", "does inspection guarantee the car", "pre purchase inspection",
        "how do I book a car inspection",
    ], responses=[
        "PakWheels inspection is a 200+ point check by trained inspectors covering the engine, "
        "accident repairs, suspension, electricals, interior, AC, body and a road test; imported "
        "cars are also checked against the auction sheet. Book it on the Car Inspection page by "
        "choosing your city, car and time slot. You get a verified digital report. It describes "
        "the car's condition at that time and is not a guarantee.",
    ], context_responses=[
        "The 200+ point inspection includes an engine computer scan, paint depth checks for "
        "accident repairs, suspension, a road test, electricals, interior, AC and body frame; "
        "Japanese imports are cross-checked with the auction sheet. It takes about 45 to 90 "
        "minutes and the QR-verified report comes by SMS and email, usually within about an "
        "hour. It's an assessment at that time, not a guarantee.",
    ]),
    dict(intent="inspection_price", category="SERVICES", kb=["Car inspection prices"], questions=[
        "how much is car inspection?", "inspection charges", "inspection ki fee kitni hai",
        "price to inspect a land cruiser", "new car PDI cost", "inspection rate for alto",
    ], responses=[
        "Inspection prices depend on the car: there are separate rates for small cars, "
        "1001-2000cc cars, and SUVs/4x4s/German cars, plus a new car pre-delivery inspection. "
        "Prices change, so check the Car Inspection page for the current rates. Inspection "
        "charges are non-refundable.",
    ], context_responses=[
        "Used car inspection costs PKR 4,950 up to 1000cc, PKR 6,950 for 1001-2000cc, and "
        "PKR 9,950 for SUVs, 4x4s, jeeps and German cars. A new car pre-delivery inspection is "
        "PKR 6,900. Inspection charges are non-refundable.",
    ]),
    # General questions only; a specific city gets a Yes/No answer (gen_city_eligibility)
    dict(intent="inspection_cities", category="SERVICES", kb=["Car inspection cities"], questions=[
        "which cities do you inspect cars in", "can the inspector come to the seller's house?",
        "inspection center location", "inspection kin shehron mein hoti hai?",
        "where is pakwheels inspection available?", "list of inspection cities",
    ], responses=[
        "PakWheels inspection is available in a number of major cities, and the inspector can "
        "usually come to your preferred location, including the seller's, subject to "
        "availability. Pick your city on the Car Inspection booking form to see if it's covered.",
    ], context_responses=[
        "Inspection is available in 12 cities: Karachi, Lahore, Islamabad, Rawalpindi, Peshawar, "
        "Faisalabad, Gujranwala, Gujrat, Hyderabad, Multan, Sargodha and Sialkot. In Lahore you "
        "can use the Inspection Center or a location of your choice; elsewhere the inspector "
        "comes to your preferred location, subject to availability.",
    ]),
    dict(intent="inspection_reschedule", category="SERVICES", questions=[
        "I need to reschedule my inspection", "the car was sold before the inspection",
        "can I get a refund for inspection?", "inspection cancel karni hai", "inspector didn't show up",
    ], responses=[
        "Contact PakWheels support as soon as possible so they can reschedule according to "
        f"availability: {HUMAN}. If the car was sold before the inspection, they'll explain the "
        "options for your booking. Note that inspection charges are non-refundable.",
    ]),
    dict(intent="auction_sheet", category="SERVICES", kb=["Auction sheet verification"], questions=[
        "how do I verify an auction sheet?", "auction sheet verification", "check japanese car history",
        "auction sheet kaise check karun", "is the auction sheet fake?", "auction sheet not found",
        "what does an auction sheet show", "verify vitz auction sheet by chassis number",
    ], responses=[
        "For imported Japanese cars, open Auction Sheet Verification, enter the chassis number "
        "and your email, and pay the fee shown at checkout. The original auction sheet is sent "
        "to you by SMS or email and can include the auction grade, mileage, colour, auction date "
        "and sold price. Sheets are delivered as-is. If no record is found online, PakWheels "
        f"checks with the auction houses; for help call {HELPLINE}.",
    ]),
    dict(intent="registration_transfer", category="SERVICES", kb=["Car registration and ownership transfer service"], questions=[
        "can pakwheels transfer my car?", "car ownership transfer service", "register my new car",
        "documents for car transfer", "how long does transfer take",
        "car transfer service kaise kaam karti hai?",
    ], responses=[
        "PakWheels offers car registration and ownership transfer services in Lahore, Karachi "
        "and Islamabad. Apply on the Car Registration or Car Transfer page; a service advisor "
        "calls to confirm the details and documents needed, you can track progress online, and "
        "the advisor arranges delivery of your documents. Exact documents and timelines depend "
        "on the province's excise office.",
    ]),
    dict(intent="car_finance", category="SERVICES", kb=["Car finance"], questions=[
        "can I buy a car on installments?", "car loan", "car finance calculator",
        "qiston par gari", "which bank gives car loan", "down payment for car finance",
        "car lease kaise milegi",
    ], responses=[
        "Yes. The PakWheels Car Finance page has a loan calculator that compares partner banks' "
        "rates, markup and insurance for new and used cars, and you can apply there so your "
        "request goes to partner banks. The bank decides eligibility, down payment and final "
        "terms.",
    ], context_responses=[
        "Use the Car Finance calculator to compare partner banks' rates, markup and insurance, "
        "then apply on PakWheels and your request is shared with the banks. The page's FAQ lists "
        "a minimum down payment of 35%; final terms, eligibility and documents are decided by "
        "the bank.",
    ]),
    dict(intent="car_insurance", category="SERVICES", kb=["Car insurance"], questions=[
        "car insurance quote", "can pakwheels insure my car?", "insurance karwani hai",
        "which insurance companies", "do you charge for insurance", "insurance for imported car",
    ], responses=[
        "Yes. The PakWheels Car Insurance page compares quotes from partner insurers for new, "
        "used and imported cars. PakWheels doesn't charge customers for the insurance service; "
        "you pay the insurer's premium. Service is offered in Lahore, Karachi and Islamabad, with "
        "nationwide coverage through partners.",
    ]),
    dict(intent="verification_tools", category="SERVICES", kb=["Tools for buyers"], questions=[
        "how to verify a car by number plate?", "MTMIS check", "verify driving license",
        "DLIMS license check", "gari ki verification online", "check vehicle registration online",
    ], responses=[
        "PakWheels has pages for MTMIS online vehicle verification and DLIMS driving licence "
        "verification under the More menu. They help you check registration details through the "
        "official provincial systems before buying a car.",
    ]),
    dict(intent="fuel_prices", category="SERVICES", questions=[
        "current petrol price", "diesel rate today", "petrol ki qeemat kya hai",
        "cng price", "petrol price in pakistan",
    ], responses=[
        "Fuel prices change regularly, so I don't quote them. The Current Petrol Prices page on "
        "PakWheels (under More) shows the latest petrol, diesel and CNG prices.",
    ]),
    dict(intent="other_services", category="SERVICES", kb=["Other PakWheels services and community"], questions=[
        "do you have a car service center?", "car import service", "can I buy a car battery from pakwheels",
        "pakwheels auto show", "car maintenance near me", "import a car from japan",
    ], responses=[
        "PakWheels also offers Service Centers for maintenance and repair, a Car Import service, "
        "car batteries, and Auto Show events. You'll find them under the More menu on the "
        "website, where you can see locations, details and how to book.",
    ]),
    # -------------------------------------------------------------- autostore
    dict(intent="autostore_order", category="AUTOSTORE", kb=["AutoStore payment methods"], questions=[
        "how do I buy car parts?", "do you have cash on delivery?", "payment methods for autostore",
        "can I pay with easypaisa", "order car mats online", "autostore kya hai",
    ], responses=[
        "PakWheels AutoStore sells parts and accessories online. Find the item under Auto Store, "
        "add it to your cart and check out. Payment options include cash on delivery, JazzCash, "
        "EasyPaisa and Visa, MasterCard or UnionPay cards.",
    ]),
    dict(intent="autostore_return", category="AUTOSTORE", kb=["AutoStore returns and refunds"], questions=[
        "how do I return a product?", "received wrong car mats", "item damaged on delivery",
        "can I return if I changed my mind?", "return policy", "product wapas karna hai",
        "return window kitne din ka hai",
    ], responses=[
        "AutoStore accepts returns for damaged, defective, incomplete or wrong items, or items "
        "that don't match the description; change of mind isn't accepted for car or bike "
        "accessories. Start the return quickly after delivery, as the window is short, by "
        f"contacting {HUMAN} with your order details and photos.",
    ], context_responses=[
        "Returns are accepted for damaged, defective, incomplete or wrong items, or items that "
        "don't match the listing, but not for change of mind on accessories. Start the return "
        "within 3 business days (72 hours) of receiving the order. Replacements ship in 5 to 10 "
        "working days, and approved refunds are paid by bank deposit within 7 to 10 working days.",
    ]),
    dict(intent="order_status", category="AUTOSTORE",
         near=["AutoStore returns and refunds", "AutoStore payment methods"], questions=[
        "can you tell me where my order is?", "my refund hasn't arrived", "order kab aayega?", "track my autostore order",
        "I returned the item but no refund yet", "order status check karna hai",
        "my autostore order hasn't arrived yet", "when will my car cover be delivered",
        "refund kab milega?",
    ], responses=[
        "I can't see order or refund details from here. Please contact "
        f"{HUMAN} with your order number. For approved returns, replacements usually ship "
        "within 5 to 10 working days and refunds are paid by bank deposit within 7 to 10 "
        "working days after approval.",
    ]),
    # -------------------------------------------------------------- community
    dict(intent="forums", category="COMMUNITY", questions=[
        "my forum post was deleted", "what images can I upload on the forum", "forum topic removed why",
        "pakwheels forum kaise use karun",
    ], responses=[
        "Forum posts can be moved to the right section or removed if they break the terms of "
        "service (for example spam or explicit content). Forums accept JPEG, PNG and GIF images.",
    ]),
    dict(intent="write_review", category="COMMUNITY", questions=[
        "how do I write a car review?", "review my car on pakwheels", "owner review kaise likhun",
        "rate my car",
    ], responses=[
        "Sign in, go to New Cars and open Reviews, then choose Write a Review, fill in the form "
        "and submit.",
    ]),
    dict(intent="cool_rides", category="COMMUNITY", kb=["Other PakWheels services and community"], questions=[
        "how do I add my car to cool rides?", "submit my ride", "can I post my bike in rides",
        "my ride submission was rejected",
    ], responses=[
        "Sign in, open Cool Rides and add your ride with real photos of your car or bike "
        "(catalogue or brochure pictures aren't accepted). Submissions are reviewed and appear "
        "once approved; they can be rejected for incomplete details, non-original photos or "
        "content against the terms.",
    ]),
    dict(intent="app_language", category="GENERAL", kb=["Other PakWheels services and community"], questions=[
        "is there a pakwheels app?", "download app", "urdu mein website hai?", "app for iphone",
        "huawei phone app",
    ], responses=[
        "Yes. The PakWheels app is available on Google Play, the App Store and Huawei "
        "AppGallery, and the website also has an Urdu version (use the اردو link at the top).",
    ]),
    # ---------------------------------------------------------- boundaries
    dict(intent="out_of_scope", category="OUT_OF_SCOPE", questions=[
        "what's the weather in lahore?", "who won the cricket match?", "write me a python script",
        "share a good biryani recipe", "help with my math homework", "who should I vote for",
        "tell me a joke about politicians", "what is the capital of france", "book a flight for me",
        # The model translated text when asked (fresh test F5): requests it can do, but shouldn't
        "translate 'thank you very much' into Spanish", "how do you say 'car' in German?",
        "write an email to my boss asking for two days' leave", "write a poem about the sea",
        "help me write my CV", "solve 25 times 17", "explain how black holes work",
        "write a SQL query to find duplicate rows", "suggest a netflix series to watch",
        "what's the best phone under 50,000 rupees?", "give me a weekly workout plan",
        "how do I lose 5 kg in a month?", "what's the bitcoin price?", "book a hotel in murree for me",
        "who is the prime minister of pakistan?", "summarize the history of the Mughal empire",
        "English mein translate kar do: mera naam Ali hai", "mujhe ek kahani sunao",
        "cricket ka score batao", "mere liye birthday wish likh do", "aaj mausam kaisa hai?",
        "recommend a good restaurant in islamabad",
    ], responses=[
        "Sorry, I can only help with cars, bikes, auto parts and PakWheels services. Is there "
        "something about buying, selling or your PakWheels account I can help with?",
    ]),
    dict(intent="unsafe_request", category="OUT_OF_SCOPE", ground_all=True,
         near=["Car inspection coverage and process", "Auction sheet verification",
               "Car registration and ownership transfer service", "Inspected vs Certified cars"],
         questions=[
        "how do I roll back the odometer?", "can you make a fake auction sheet",
        "how to change chassis number", "sell a stolen car without papers",
        "meter kam kaise karte hain", "edit an inspection report",
        "how can I lower the mileage shown on my car's meter?", "odometer reverse karna hai",
        "how to tamper with the speedometer reading", "can you help me fake the registration papers",
        "how do I hide an accident from the inspector?", "change the engine number on my car",
        "make me a fake token tax receipt", "how to reset the km reading before selling",
    ], responses=[
        "I can't help with that. Tampering with odometers, chassis numbers or documents is "
        "illegal and harms buyers. I'm happy to help you sell or buy a car the right way, for "
        "example with an inspection report or a verified auction sheet.",
        "I can't help with that. Changing a car's mileage, numbers, documents or history is "
        "illegal and misleads buyers. If you're selling, an honest ad with a PakWheels inspection "
        "report builds more trust; if you're buying, an inspection can help spot tampering.",
    ]),
    dict(intent="personal_data_request", category="OUT_OF_SCOPE", ground_all=True,
         near=["Account, login and privacy", "Contact and support hours"], questions=[
        "give me the seller's phone number for ad 12345", "who owns this car number LEA-1234",
        "send me the buyer's address", "share the email of this seller",
    ], responses=[
        "I can't share other users' personal details. To reach a seller, use the contact "
        "options on their ad. To check a vehicle's registration, use the official MTMIS "
        "verification for its province.",
    ]),
]

# Questions whose answer is NOT in the given context (retrieval miss): the
# model must not borrow figures from an unrelated section.
CONTEXT_GAPS = [
    ("Is Sell It For Me available in Multan?", "Sell It For Me eligibility and process",
     "No, Multan isn't on the current Sell It For Me list (Karachi, Lahore, Faisalabad, Gujranwala, "
     "Islamabad, Rawalpindi and Peshawar). You can still post a free ad yourself, and feature "
     "it for more visibility."),
    ("Can I return car mats if they don't match my taste?", "AutoStore returns and refunds",
     "No. Change of mind isn't accepted for car or bike accessories. Returns are "
     "accepted for damaged, defective, incomplete or wrong items, or items that don't match the "
     "description, within 3 business days of delivery."),
    ("Is human support available at 11 pm?", "Contact and support hours",
     f"No, human agents are available {HOURS}, so not at 11 pm. You can leave a message through "
     "the Contact Us form or email info@pakwheels.com, and I can answer general questions now."),
]


# --------------------------------------------------------------------------
# v2 generators: question-specific answers grounded in one KB section
# --------------------------------------------------------------------------

# (car, tier); tiers follow the inspection / Sell It For Me price bands
CARS = [
    ("Suzuki Alto", "small"), ("Suzuki Mehran", "small"), ("Suzuki Cultus", "small"),
    ("Suzuki Wagon R", "small"), ("Daihatsu Mira", "small"), ("Nissan Dayz", "small"),
    ("Toyota Passo", "small"), ("Daihatsu Cuore", "small"),
    ("Toyota Corolla", "mid"), ("Honda Civic", "mid"), ("Honda City", "mid"),
    ("Toyota Yaris", "mid"), ("Suzuki Swift", "mid"), ("Toyota Aqua", "mid"),
    ("Toyota Prius", "mid"), ("Hyundai Elantra", "mid"), ("Changan Alsvin", "mid"),
    ("Toyota Land Cruiser", "suv"), ("Toyota Prado", "suv"), ("Toyota Fortuner", "suv"),
    ("Kia Sportage", "suv"), ("Hyundai Tucson", "suv"), ("Haval H6", "suv"),
    ("Mercedes C-Class", "german"), ("BMW 3 Series", "german"), ("Audi A4", "german"),
]
# Engine sizes across both bands and their edges: with only a few, the model
# memorised answers and put a 1300cc car in the "up to 1000cc" band.
CC_TIERS = [(600, "small"), (660, "small"), (800, "small"), (850, "small"), (997, "small"),
            (1000, "small"), (1050, "mid"), (1197, "mid"), (1200, "mid"), (1250, "mid"),
            (1298, "mid"), (1340, "mid"), (1490, "mid"), (1500, "mid"), (1598, "mid"),
            (1800, "mid"), (1990, "mid"), (2000, "mid")]
# (service, cc) pairs asked in scripts/demo_eval.py stay out of training
DEMO_CC = {("inspection", 1000), ("sifm", 1300)}

NON_INSPECTION_CITIES = ["Quetta", "Bahawalpur", "Abbottabad", "Sukkur", "Mardan", "Sahiwal",
                         "Rahim Yar Khan", "Jhelum", "Gilgit", "Mirpur"]
NON_SIFM_CITIES = ["Multan", "Sialkot", "Hyderabad", "Gujrat", "Sargodha", "Quetta",
                   "Bahawalpur", "Abbottabad"]
NON_TRANSFER_CITIES = ["Rawalpindi", "Peshawar", "Faisalabad", "Multan", "Quetta", "Hyderabad",
                       "Sialkot"]


def gen_inspection_price(rng):
    templates = ["What does PakWheels charge to inspect a {car}?", "{car} ki inspection fee kitni hai?",
                 "Inspection price for my {car}?", "I want a used {car} inspected, how much will it cost?"]
    for car, tier in CARS:
        for t in rng.sample(templates, 2):
            yield (t.format(car=car), inspection_answer(car_phrase(car, tier), tier),
                   "Car inspection prices")
    cc_templates = ["Inspection charges for a {cc}cc car?", "{cc}cc gari ki inspection fee?",
                    "How much to inspect my {cc}cc car?"]
    for cc, tier in CC_TIERS:
        if ("inspection", cc) in DEMO_CC:
            continue
        for t in rng.sample(cc_templates, 2):
            yield (t.format(cc=cc), inspection_answer(cc_phrase(cc, tier), tier),
                   "Car inspection prices")
    for q in ["Inspection fee for a 2500cc sedan?", "How much to inspect a 3000cc Camry?"]:
        yield (q, LARGE_INSPECTION_ANSWER, "Car inspection prices")


def gen_sifm_fee(rng):
    templates = ["Sell It For Me charges for my {car}?",
                 "{car} Sell It For Me se bechni hai, fee kitni hogi?",
                 "What is the onboarding fee to sell a {car} through Sell It For Me?"]
    for car, tier in CARS:
        yield (rng.choice(templates).format(car=car), sifm_fee_answer(car_phrase(car, tier), tier),
               "Sell It For Me fees")
    cc_templates = ["Sell It For Me fee for a {cc}cc car?",
                    "{cc}cc car Sell It For Me se bechni hai, kitni fee?",
                    "What's the Sell It For Me onboarding fee for a {cc}cc car?"]
    for cc, tier in CC_TIERS:
        if ("sifm", cc) in DEMO_CC:
            continue
        for t in rng.sample(cc_templates, 2):
            yield (t.format(cc=cc), sifm_fee_answer(cc_phrase(cc, tier), tier), "Sell It For Me fees")
    for q in ["Sell It For Me fee for a 2500cc Camry?", "Sell It For Me charges for a 3000cc sedan?"]:
        yield (q, LARGE_SIFM_ANSWER, "Sell It For Me fees")


def gen_commission(rng):
    templates = ["If my car sells for {p} through Sell It For Me, how much commission do I pay?",
                 "Gari {p} mein biki to Sell It For Me ka commission kitna hoga?",
                 "Sell It For Me commission if I sell for {p}?",
                 "My car sold for {p} via Sell It For Me. What's PakWheels' cut?"]
    # As many prices at or under 5 lakh as above it: with few, the model applied
    # 1% to 4 lakh and skipped the minimum. 4 lakh itself is a demo question.
    for lakh in [1.5, 2, 2.5, 3, 3.5, 4.2, 4.5, 4.8, 5, 6, 8, 12, 18.5, 25, 40, 65, 100]:
        n_rows = 2 if lakh <= 5 else 1
        for t in rng.sample(templates, n_rows):
            yield (t.format(p=lakh_label(lakh)), commission_answer(lakh), "Sell It For Me fees")


def gen_featured_price(rng):
    templates = ["How much to feature my ad for {d}?", "{d} ke liye ad feature karne ka kharcha?",
                 "Price of a featured ad for {d}?"]
    for label in FEATURED:
        yield (rng.choice(templates).format(d=label), featured_answer(label), "Featured ad prices")
    for n in (5, 10):
        yield (f"Price for a bundle of {n} feature credits?", bundle_answer(n), "Featured ad prices")


def gen_city_eligibility(rng):
    insp_q = ["Can I get a car inspected in {c}?", "{c} mein car inspection hoti hai?",
              "Do you inspect cars in {c}?", "{c} mein gari check karwa sakte hain?",
              "Kya {c} mein inspector aa sakta hai?", "{c} wale inspection book kar sakte hain?"]
    sifm_q = ["Is Sell It For Me available in {c}?", "{c} mein Sell It For Me hai?",
              "Can PakWheels sell my car for me in {c}?", "{c} se Sell It For Me ke through gari bik sakti hai?",
              "Kya {c} mein Sell It For Me ki service milti hai?", "I'm in {c}, does Sell It For Me work here?"]
    for c in INSPECTION_CITIES + NON_INSPECTION_CITIES:
        yield (rng.choice(insp_q).format(c=c), inspection_city_answer(c), "Car inspection cities")
    for c in SIFM_CITIES + NON_SIFM_CITIES:
        yield (rng.choice(sifm_q).format(c=c), sifm_city_answer(c),
               "Sell It For Me eligibility and process")


def gen_transfer_city(rng):
    # "Can PakWheels ... my car in {city}?" was answered as Sell It For Me; the same
    # sentence shape needs its own service's Yes/No
    templates = ["Does PakWheels do car transfers in {c}?", "{c} mein car transfer karwa sakte hain?",
                 "Can you register my new car in {c}?", "Is the ownership transfer service available in {c}?",
                 "Can PakWheels handle my car's transfer in {c}?"]
    for c in TRANSFER_CITIES:
        for t in rng.sample(templates, 3):
            yield (t.format(c=c), transfer_city_answer(c), "Car registration and ownership transfer service")
    for c in NON_TRANSFER_CITIES:
        yield (rng.choice(templates).format(c=c), transfer_city_answer(c),
               "Car registration and ownership transfer service")


def gen_model_year(rng):
    templates = ["Can I sell my {y} {car} through Sell It For Me?", "Is a {y} {car} eligible for Sell It For Me?",
                 "{y} model {car} Sell It For Me mein chalegi?"]
    for car, year in [("Toyota Corolla", 1995), ("Honda Civic", 1999), ("Suzuki Mehran", 1997),
                      ("Toyota Land Cruiser", 1992), ("Honda City", 2000), ("Suzuki Cultus", 2004),
                      ("Toyota Corolla", 2012), ("Honda Civic", 2018), ("Suzuki Alto", 2021)]:
        yield (rng.choice(templates).format(y=year, car=car), model_year_answer(year, car),
               "Sell It For Me eligibility and process")
    for q in ["My car is on an open letter, can I use Sell It For Me?",
              "open letter gari Sell It For Me se bik sakti hai?",
              "Does Sell It For Me accept open letter cars?"]:
        yield (q, OPEN_LETTER_ANSWER, "Sell It For Me eligibility and process")


def gen_auction_fee(rng):
    for q in ["What is the fee for auction sheet verification?", "auction sheet check karne ke kitne paise?",
              "Price to verify an auction sheet?", "Auction sheet report charges?",
              "How much do you charge for a Vitz auction sheet?"]:
        yield (q, AUCTION_FEE_ANSWER, "Auction sheet verification")


def gen_autostore_returns(rng):
    issues = [("the wrong {i}", "wrong"), ("a damaged {i}", "damaged"), ("a broken {i}", "defective"),
              ("an incomplete {i} (parts missing)", "incomplete")]
    items = ["car cover", "dash cam", "seat cover set", "pair of wiper blades", "steering cover",
             "set of LED lights"]
    templates = ["I received {x}, what do I do?", "Order mein {x} aaya hai, return kaise karun?",
                 "Got {x} from AutoStore, can I return it?"]
    for item in items:
        phrase, kind = rng.choice(issues)
        x = phrase.format(i=item)
        yield (rng.choice(templates).format(x=x),
               f"Yes, a {kind} item qualifies for a return. Start it within 3 business days (72 "
               f"hours) of delivery by contacting {HUMAN} with your order number and photos. "
               "Replacements ship in 5 to 10 working days, and approved refunds are paid by bank "
               "deposit within 7 to 10 working days.", "AutoStore returns and refunds")
    for item in ["floor mats", "seat covers", "air freshener", "phone holder"]:
        yield (f"Can I return {item} if I just don't like them?",
               "No. Change of mind isn't accepted for car or bike accessories. You can return an item "
               "only if it's damaged, defective, incomplete, wrong or doesn't match the listing, "
               "within 3 business days of delivery.", "AutoStore returns and refunds")
    # "I don't like the colour" is change of mind; "a different colour arrived" is a wrong item
    dislike = ["Can I send back a {i} because I don't like its colour?",
               "{i} ka colour pasand nahi aaya, wapas ho sakta hai?",
               "The {i} looks different from what I imagined, can I return it?"]
    for item, t in zip(["steering cover", "seat cover set", "car cover"], dislike):
        yield (t.format(i=item),
               "No. Not liking the colour or look is a change of mind, which isn't accepted for car "
               "or bike accessories. A colour return is accepted only if you received a different "
               "colour from the one you ordered, reported within 3 business days of delivery.",
               "AutoStore returns and refunds")
    for item, ordered, got in [("seat cover set", "black", "beige"), ("car cover", "silver", "blue"),
                               ("steering cover", "red", "black")]:
        yield (f"I ordered a {ordered} {item} but received a {got} one.",
               f"Yes, a different colour from the one you ordered counts as a wrong item, so it "
               f"qualifies for a return. Start it within 3 business days (72 hours) of delivery by "
               f"contacting {HUMAN} with your order number and photos.",
               "AutoStore returns and refunds")


def _named(car_or_cc):
    return car_or_cc if isinstance(car_or_cc, str) else f"{car_or_cc}cc car"


def _lead(car_or_cc, tier):
    return (car_phrase(car_or_cc, tier) if isinstance(car_or_cc, str)
            else cc_phrase(car_or_cc, tier))


FOLLOW_UP_CC = [(660, "small"), (997, "small"), (1250, "mid"), (1600, "mid"), (2000, "mid")]
FOLLOW_UP_SUBJECT = ["And for my {x}?", "What about my {x}?", "aur {x} ke liye?", "Same for my {x}?"]
FOLLOW_UP_CITY = ["And in {c}?", "What about {c}?", "aur {c} mein?"]
FOLLOW_UPS_PER_SERVICE = 10


def gen_follow_ups(rng):
    """
    Two-turn chats: a specific answer, then a short follow-up about another car,
    city or duration. The answer is for the same service as the first question.
    Yields (history, question, answer, KB title).
    """
    subjects = CARS + [(cc, tier) for cc, tier in FOLLOW_UP_CC]
    services = [
        ("Car inspection prices", inspection_answer,
         ["What does an inspection cost for my {x}?", "{x} ki inspection kitne ki hai?"]),
        ("Sell It For Me fees", sifm_fee_answer,
         ["Sell It For Me fee for my {x}?", "{x} Sell It For Me se bechne ki fee?"]),
    ]
    for title, answer, first_templates in services:
        for _ in range(FOLLOW_UPS_PER_SERVICE):
            (a, tier_a), (b, tier_b) = rng.sample(subjects, 2)
            first = rng.choice(first_templates).format(x=_named(a))
            history = [{"role": "user", "content": first},
                       {"role": "assistant", "content": answer(_lead(a, tier_a), tier_a)}]
            yield (history, rng.choice(FOLLOW_UP_SUBJECT).format(x=_named(b)),
                   answer(_lead(b, tier_b), tier_b), title)

    cities = [("Car inspection cities", inspection_city_answer,
               INSPECTION_CITIES + NON_INSPECTION_CITIES, "Can I book a car inspection in {c}?"),
              ("Sell It For Me eligibility and process", sifm_city_answer,
               SIFM_CITIES + NON_SIFM_CITIES, "Does Sell It For Me work in {c}?"),
              ("Car registration and ownership transfer service", transfer_city_answer,
               TRANSFER_CITIES + NON_TRANSFER_CITIES, "Is car transfer available in {c}?")]
    for title, answer, pool, first_template in cities:
        for _ in range(FOLLOW_UPS_PER_SERVICE):
            a, b = rng.sample(pool, 2)
            history = [{"role": "user", "content": first_template.format(c=a)},
                       {"role": "assistant", "content": answer(a)}]
            yield (history, rng.choice(FOLLOW_UP_CITY).format(c=b), answer(b), title)

    for a, b in [("7 days", "14 days"), ("14 days", "a month"), ("28 days", "a week")]:
        history = [{"role": "user", "content": f"How much to feature my ad for {a}?"},
                   {"role": "assistant", "content": featured_answer(a)}]
        yield (history, f"And for {b}?", featured_answer(b), "Featured ad prices")


GENERATORS = [
    ("inspection_price_specific", "SERVICES", gen_inspection_price),
    ("sifm_fee_specific", "SELLING", gen_sifm_fee),
    ("sifm_commission", "SELLING", gen_commission),
    ("featured_price_specific", "SELLING", gen_featured_price),
    ("city_eligibility", "SERVICES", gen_city_eligibility),
    ("transfer_city", "SERVICES", gen_transfer_city),
    ("sifm_model_year", "SELLING", gen_model_year),
    ("auction_fee_unknown", "SERVICES", gen_auction_fee),
    ("autostore_return_specific", "AUTOSTORE", gen_autostore_returns),
]


def _demo_questions() -> set:
    """The demo questions are kept out of training, so the demo measures generalisation."""
    from scripts.demo_eval import CASES
    return {c["q"].strip().lower() for c in CASES}


def build(seed: int = SEED) -> list:
    rng = random.Random(seed)
    kb = sections_by_title(KB_PATH)
    titles = list(kb)
    rows = []
    # Single-turn rows that answer their own question (no wrong-section or
    # retrieval-miss rows): the pool for topic-switch chats
    switch_pool = []
    overridden = []  # (question, old response): answers the verified answer corrected

    def add(intent, question, response, context=None, history=None):
        # The verified answer the app would add for this question goes first in the
        # context; a response that doesn't contain it is replaced by it
        facts = detect(question, history)
        if facts:
            verified = " ".join(f.text for f in facts)
            context = f"{FACT_HEADER}\n{verified}" + (f"\n---\n{context}" if context else "")
            if verified not in response:
                overridden.append((question, response))
                response = verified
        row = {
            "instruction": question,
            "response": response,
            "context": context or "",
            "history": history or [],
            "category": intent["category"],
            "intent": intent["intent"],
        }
        rows.append(row)
        return row

    def distractor(own_titles):
        """A look-alike section most of the time, otherwise any other section."""
        confusable = [t for o in own_titles for t in CONFUSABLE.get(o, []) if t not in own_titles]
        if confusable and rng.random() < CONFUSABLE_DISTRACTOR_RATE:
            return rng.choice(confusable)
        return rng.choice([t for t in titles if t not in own_titles])

    def grounded_context(own_titles, hard=None):
        """The right section(s), often with a distractor. `hard`: the look-alike section
        retrieval returns alongside this one, used for most distractors."""
        chunks = [kb[t] for t in own_titles]
        if rng.random() < 0.6:
            chunks.append(kb[hard if hard and rng.random() < HARD_DISTRACTOR_RATE
                             else distractor(own_titles)])
            rng.shuffle(chunks)
        return "\n---\n".join(chunks)

    for intent in INTENTS:
        own = intent.get("kb", [])
        near = intent.get("near", [])
        for title in own + near:
            if title not in kb:
                raise KeyError(f"{intent['intent']}: KB section '{title}' not found")

        # 1. Every phrasing without context
        for q in intent["questions"]:
            row = add(intent, q, rng.choice(intent["responses"]))
            if not own:
                switch_pool.append((row, None))

        if not own:
            # 1b. No KB section, but retrieval often returns one anyway (an advance-payment
            #     scam question retrieves "Sell It For Me fees"): answer as without context
            questions = (intent["questions"] if intent.get("ground_all") else
                         rng.sample(intent["questions"],
                                    min(IRRELEVANT_CONTEXT_EXAMPLES_PER_INTENT, len(intent["questions"]))))
            for q in questions:
                pool = near or titles
                chunks = rng.sample(pool, min(len(pool), rng.choice([1, 2])))
                switch_pool.append((add(intent, q, rng.choice(intent["responses"]),
                                        "\n---\n".join(kb[t] for t in chunks)), None))
            continue

        # 2. Grounded rows: every phrasing for intents with figures or marked
        #    ground_all (e.g. payment: the right section must win), a sample otherwise
        ctx_responses = intent.get("context_responses", intent["responses"])
        ground_all = "context_responses" in intent or intent.get("ground_all")
        questions = (intent["questions"] if ground_all else
                     rng.sample(intent["questions"],
                                min(CONTEXT_EXAMPLES_PER_INTENT, len(intent["questions"]))))
        for q in questions:
            context = grounded_context(own, intent.get("hard_distractor"))
            switch_pool.append((add(intent, q, rng.choice(ctx_responses), context), own[0]))

        # 3. Wrong section only: a look-alike section was retrieved, the right one wasn't.
        #    Answer from the intent and ignore the retrieved figures.
        for q in rng.sample(intent["questions"],
                            min(WRONG_SECTION_EXAMPLES_PER_INTENT, len(intent["questions"]))):
            add(intent, q, rng.choice(intent["responses"]), kb[distractor(own)])

    # 4. Retrieval misses: fee questions with an unrelated section -> no figures
    fee_intents = [i for i in INTENTS if "context_responses" in i]
    for _ in range(RETRIEVAL_MISS_EXAMPLES):
        intent = rng.choice(fee_intents)
        unrelated = rng.choice([t for t in titles if t not in intent["kb"]])
        add(intent, rng.choice(intent["questions"]), rng.choice(intent["responses"]), kb[unrelated])

    # 5. Questions the context answers only partly
    gap_intent = {"category": "CONTEXT_GAP", "intent": "context_gap"}
    for question, title, response in CONTEXT_GAPS:
        add(gap_intent, question, response, kb[title])

    # 6. Question-specific answers (v2)
    for name, category, generator in GENERATORS:
        intent = {"category": category, "intent": name}
        for question, response, title in generator(rng):
            switch_pool.append((add(intent, question, response, grounded_context([title])), title))

    # 7. Follow-ups (v3): same service, new car / city / duration
    follow_up = {"category": "FOLLOW_UP", "intent": "follow_up"}
    for history, question, response, title in gen_follow_ups(rng):
        add(follow_up, question, response, grounded_context([title]), history)

    # 8. Topic switches (v3): earlier exchanges about something else, often a
    #    look-alike section, must not leak into the answer to the new question
    grounded = [(r, t) for r, t in switch_pool if t]
    no_context = [(r, t) for r, t in switch_pool if not t]
    for _ in range(TOPIC_SWITCH_EXAMPLES):
        current, title = rng.choice(no_context if rng.random() < TOPIC_SWITCH_NO_CONTEXT_RATE
                                    else grounded)
        others = [(r, t) for r, t in grounded if r["intent"] != current["intent"]]
        look_alike = [(r, t) for r, t in others if t in CONFUSABLE.get(title, [])]
        n_previous = 2 if rng.random() < TWO_PREVIOUS_TURNS_RATE else 1
        history = []
        for previous, _ in rng.sample(look_alike if look_alike else others, n_previous):
            history += [{"role": "user", "content": previous["instruction"]},
                        {"role": "assistant", "content": previous["response"]}]
        add(current, current["instruction"], current["response"],
            strip_fact_block(current["context"]), history)

    demo = _demo_questions()
    asked = [r["instruction"] for r in rows]
    asked += [m["content"] for r in rows for m in r["history"] if m["role"] == "user"]
    leaked = sorted({q for q in asked if q.strip().lower() in demo})
    if leaked:
        raise ValueError(f"Demo questions must not be training rows: {leaked}")

    build.overridden = overridden  # for main() to report
    rng.shuffle(rows)
    return rows


def main():
    rows = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with_ctx = sum(1 for r in rows if r["context"])
    verified = sum(1 for r in rows if r["context"].startswith(FACT_HEADER))
    print(f"Wrote {len(rows)} rows ({with_ctx} with context, {verified} with a verified answer, "
          f"{len({r['intent'] for r in rows})} intents) to {OUT_PATH}")
    print(f"{len(build.overridden)} answers replaced by the verified answer")


if __name__ == "__main__":
    main()
