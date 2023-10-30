#!/bin/bash


g++ main.cpp matrix.cpp multiply.cpp -std=c++1z -pthread -mfma -o \
    main -D JUDGE_RIGHT -D N=280 -D M=8 -D P=124

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "测试计算编译成功，可执行文件 'main' 已生成。"
else
    echo "测试计算编译失败，请检查源代码文件和编译选项。"
    exit 1
fi

./main

g++ main.cpp matrix.cpp multiply.cpp -std=c++1z -pthread -mfma -o \
    main -D N=1024 -D M=1024 -D P=1024

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "测试时间编译完成，可执行文件 'main' 已生成。"
else
    echo "测试时间编译失败，请检查源代码文件和编译选项。"
    exit 1
fi

./main

rm ./main

