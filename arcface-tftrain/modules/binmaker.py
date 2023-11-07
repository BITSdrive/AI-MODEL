import bcolz

# 데이터 로드
carray = bcolz.open('/home/bits/arcface-tf2/data/test_dataset/kface1/kface1')

# carray를 NumPy 배열로 변환
numpy_array = carray[:]

print(numpy_array[-2:])