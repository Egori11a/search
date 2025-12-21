docker run -d --name mongo -p 27017:27017 -v mongo_data:/data/db mongo:7
docker ps
pip install pyyaml requests pymongo beautifulsoup4
python crawler.py config.yaml

docker start mongo
docker stop mongo

Проверка в MongoDB
sudo apt-get install -y mongosh
Зайди:
mongosh
Команды проверки:
use crawler_db
// сколько URL в очереди
db.urls.countDocuments()
// сколько документов реально сохранено
db.docs.countDocuments()
// посмотреть пару записей
db.urls.find().limit(3).pretty()
db.docs.find({}, {url:1, source:1, crawl_ts:1, _id:0}).limit(5).pretty()
// проверить, что есть планирование переобкачки
db.urls.find({}, {url:1, next_crawl_ts:1, status_code:1, _id:0}).limit(5).pretty()
// посмотреть, есть ли новые URL, которые нашли из ссылок
db.urls.find({discovered_from: {$ne: null}}, {url:1, discovered_from:1, _id:0}).limit(5).pretty()
Что считается “правильной работой”
db.urls.countDocuments() ≈ 2000 (твои 1000+1000) и может расти, если найдёт новые рецепты по ссылкам.
db.docs.countDocuments() растёт, когда он впервые обходит URL и/или когда контент меняется.
После остановки Ctrl+C и повторного запуска он не начинает заново, а продолжает по next_crawl_ts.
Важное замечание (про объём)
MongoDB имеет лимит 16MB на один документ. Если какая-то страница будет огромной — insert_one упадёт. Для рецептов обычно норм, но если вдруг будет ошибка — скажешь, я добавлю “сжатие” или сохранение raw_html в файл + путь в Mongo.