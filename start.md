Запуск монго: sudo systemctl start mongod
Проверка монго: sudo systemctl status mongod
Зависимости питона: python3 -m pip install --user pyyaml requests pymongo beautifulsoup4 tqdm

ЛР2:
docker run -d --name mongo -p 27017:27017 -v mongo_data:/data/db mongo:7
docker ps
pip install pyyaml requests pymongo beautifulsoup4
docker exec -it mongo mongosh crawler_db --eval 'db.docs.dropIndex("url_1")'
docker exec -it mongo mongosh crawler_db --eval 'db.docs.createIndex({url:1},{unique:true,name:"url_1"})'
python crawler.py config.yaml

docker start mongo
docker stop mongo

Проверка в MongoDB
sudo apt-get install -y mongosh
mongosh

Внутри
"""use crawler_db
db.docs.countDocuments()
db.docs.find({}, {url:1, source:1, crawl_ts:1, title:1}).limit(3).pretty()"""

ЛР3:
cd /home/egor/search/345
mkdir -p build
cd build
cmake .. && make -j
./lab3_token_stats ../config.yaml

ЛР4:
cd /home/egor/search/345/build
./lab4_zipf ../config.yaml

ls -la zipf.csv
head zipf.csv

ЛР5:
cd /home/egor/search/345/build
./lab5_search ../config.yaml

./lab5_search ../config.yaml --eval ../queries_test.tsv

ЛР6:
cd /home/egor/search/6
mkdir -p build
cd build
cmake .. && make -j

./lab6_build_index ../config.yaml index.bidx

./lab6_bool_search index.bidx

уха
курица AND помидоры
курица NOT салат
курица OR котлеты

ЛР7:
cd /home/egor/search/7
mkdir -p build
cd build
cmake .. && make -j

./lab7_cli /home/egor/search/6/build/index.bidx /home/egor/search/7/queries7.txt

./lab7_server /home/egor/search/6/build/index.bidx 8080

московский авиационный институт
(красный || желтый) автомобиль
руки !ноги