#!/usr/bin/env bash

source venv/bin/activate

mkdir -p logs

printf '\nProcessing All Categories...\n'
python analyze.py reviews > logs/All_Categories.log 2> /dev/null

printf '\nProcessing Amazon Instant Video...\n'
python analyze.py reviews/reviews_Amazon_Instant_Video_5.json.gz > logs/Amazon_Instant_Video.log 2> /dev/null

printf '\nProcessing Apps for Android...\n'
python analyze.py reviews/reviews_Apps_for_Android_5.json.gz > logs/Apps_for_Android.log 2> /dev/null

printf '\nProcessing Automotive...\n'
python analyze.py reviews/reviews_Automotive_5.json.gz > logs/Automotive.log 2> /dev/null

printf '\nProcessing Baby...\n'
python analyze.py reviews/reviews_Baby_5.json.gz > logs/Baby.log 2> /dev/null

printf '\nProcessing Beauty...\n'
python analyze.py reviews/reviews_Beauty_5.json.gz > logs/Beauty.log 2> /dev/null

printf '\nProcessing Books...\n'
python analyze.py reviews/reviews_Books_5.json.gz > logs/Books.log 2> /dev/null

printf '\nProcessing CDs_and_Vinyl...\n'
python analyze.py reviews/reviews_CDs_and_Vinyl_5.json.gz > logs/CDs_and_Vinyl.log 2> /dev/null

printf '\nProcessing Cell_Phones_and_Accessories...\n'
python analyze.py reviews/reviews_Cell_Phones_and_Accessories_5.json.gz > logs/Cell_Phones_and_Accessories.log 2> /dev/null

printf '\nProcessing Clothing_Shoes_and_Jewelry...\n'
python analyze.py reviews/reviews_Clothing_Shoes_and_Jewelry_5.json.gz > logs/Clothing_Shoes_and_Jewelry.log 2> /dev/null

printf '\nProcessing Digital_Music...\n'
python analyze.py reviews/reviews_Digital_Music_5.json.gz > logs/Digital_Music.log 2> /dev/null

printf '\nProcessing Electronics...\n'
python analyze.py reviews/reviews_Electronics_5.json.gz > logs/Electronics.log 2> /dev/null

printf '\nProcessing Grocery_and_Gourmet_Food...\n'
python analyze.py reviews/reviews_Grocery_and_Gourmet_Food_5.json.gz > logs/Grocery_and_Gourmet_Food.log 2> /dev/null

printf '\nProcessing Health_and_Personal_Care...\n'
python analyze.py reviews/reviews_Health_and_Personal_Care_5.json.gz > logs/Health_and_Personal_Care.log 2> /dev/null

printf '\nProcessing Home_and_Kitchen...\n'
python analyze.py reviews/reviews_Home_and_Kitchen_5.json.gz > logs/Home_and_Kitchen.log 2> /dev/null

printf '\nProcessing Kindle_Store...\n'
python analyze.py reviews/reviews_Kindle_Store_5.json.gz > logs/Kindle_Store.log 2> /dev/null

printf '\nProcessing Movies_and_TV...\n'
python analyze.py reviews/reviews_Movies_and_TV_5.json.gz > logs/Movies_and_TV.log 2> /dev/null

printf '\nProcessing Musical_Instruments...\n'
python analyze.py reviews/reviews_Musical_Instruments_5.json.gz > logs/Musical_Instruments.log 2> /dev/null

printf '\nProcessing Office_Products...\n'
python analyze.py reviews/reviews_Office_Products_5.json.gz > logs/Office_Products.log 2> /dev/null

printf '\nProcessing Patio_Lawn_and_Garden...\n'
python analyze.py reviews/reviews_Patio_Lawn_and_Garden_5.json.gz > logs/Patio_Lawn_and_Garden.log 2> /dev/null

printf '\nProcessing Pet_Supplies...\n'
python analyze.py reviews/reviews_Pet_Supplies_5.json.gz > logs/Pet_Supplies.log 2> /dev/null

printf '\nProcessing Sports_and_Outdoors...\n'
python analyze.py reviews/reviews_Sports_and_Outdoors_5.json.gz > logs/Sports_and_Outdoors.log 2> /dev/null

printf '\nProcessing Tools_and_Home_Improvemen...\n'
python analyze.py reviews/reviews_Tools_and_Home_Improvement_5.json.gz > logs/Tools_and_Home_Improvement.log 2> /dev/null

printf '\nProcessing Toys_and_Games...\n'
python analyze.py reviews/reviews_Toys_and_Games_5.json.gz > logs/Toys_and_Games.log 2> /dev/null

printf '\nProcessing Video_Games...\n'
python analyze.py reviews/reviews_Video_Games_5.json.gz > logs/Video_Games.log 2> /dev/null
