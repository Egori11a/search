cd ~/search/7
mkdir -p build
cd build
cmake ..
make -j

cat > ~/search/7/queries7.txt << 'EOF'
московский авиационный институт
(красный || желтый) автомобиль
руки !ноги
(курица && !суп) || (котлеты && помидоры)
!(уха || суп) && (курица || рыба)
EOF

cd ~/search/7/build
./lab7_cli /home/egor/search/6/build/index.bidx /home/egor/search/7/queries7.txt

cd ~/search/7/build
./lab7_server /home/egor/search/6/build/index.bidx 8080