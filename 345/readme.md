Как собрать и запустить (пошагово)
cd ~/search/345
mkdir -p build
cd build
cmake ..
make -j

Лаба 3
./lab3_token_stats ../config.yaml

Лаба 4 (Zipf CSV)
./lab4_zipf ../config.yaml
python3 ../plot_zipf.py

Лаба 5

./lab5_search ../config.yaml