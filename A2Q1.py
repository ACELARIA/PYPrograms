def capitalize(word):
    result = ""
    for i in range(len(word)):
        if i%2 == 0:
            result+=word[i].lower()  
        else:
            result+=word[i].upper() 
    return result

user_input= input("Enter a word: ")

result_word = capitalize(user_input)
print(result_word)
