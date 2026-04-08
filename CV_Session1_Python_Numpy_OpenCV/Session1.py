# ===review python===
# input
name = input()

# output
print("nama: " + name)
print(f"nama: {name}") # fstring

# selection (if, else, elif)
age = int(input()) # type cast
print(type(age)) # print type (str, number, float)
if age > 18:
    print("legal")
else: 
    print("tetot")
    
# repetition (loop) (for, while) (gaada do while)
for i in range(20):
    print(i)

a = 0
while a < 5:
    print(a)
    a = a+1 

# function
def greet(name):
    print(f"helo {name}")
    
def bye(name="unknown"): # default value
    print(f"bye {name}")

# array
# list -> bisa ganti elemen apa aja (mutable), sizenya bisa nambah terus
fruits = ["apple", "banana", "melon"]
print(fruits)
fruits.append("watermelon")
fruits.remove("apple")
fruits.pop(1)

# tuple -> gabisa diganti (immutable)
role = ("USER", "ADMIN") # kyk const

# set -> gaboleh ada dupe (unique)
numbers = [1,2,3,4,5,5,5]
print(numbers)
numbers = set(numbers)
print(numbers)
numbers = {1,2,3,4,5,5}
print(numbers)

# dictionary -> key-value pair
student = {
    "name" : "hisuy",
    "class" : "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
    "setan" :  "ooooorizki",
} # data typenya bebas
print(student["setan"])

# numpy
import numpy as np

# list di pt -> bisa beda data type
# [1, "nama", 2.0,]
# numpy array -> member harus sama data type
numList = [1,1,1,1]
numArr = np.array(numList)
print(type(numList))
print(type(numArr))

# slicing -> ngambil beberapa elemen gitu kek tokenizing
print(numArr[:3])

# shape
twoDArray = np.array([[1,2],[3,4]])
print(twoDArray.shape)

threeDArray = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(threeDArray.shape)

zeros = np.zeros((3,2))
ones = np.ones((3,2))
empties = np.empty((3,2))

# vstack & hstack
a = np.array([10,20,30])
b = np.array([40,50,60])
vAB = np.vstack((a,b)) # numpuk atas bawah (vertical)
hAB = np.hstack((a,b)) # numpuk kiri kanan (horizontal)

# openCV
import cv2 as cv

# masukin image ke code
img = cv.imread("./assets/art.jpg")
print(img.shape) # 1200,1200,3
cv.imshow("tes", img)
cv.waitKey("q")
