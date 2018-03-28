
# parameters
WINDOW_SIZE = 5
MIN_COOCCUR_VALUE = 1
COOCCURENCE_FILE = "Cooccur.csv"
WORD2ID_FILE = "Word2Id.csv"

# main data structures
cooccur_map = {}
word2id = {}

# example
text = "Prime Minister Theresa May will remind her cabinet that discussions must remain private"
text2 = "Theresa Mary May is a British politician who has served as Prime Minister"
sentences = []
sentences.append(text.lower().split())
sentences.append(text2.lower().split())

# populate word2id
id = 1
for sentence in sentences:
    for word in sentence:
        if word not in word2id:
            word2id[word] = id
            id += 1

# populate cooccur_map
for sentence in sentences:
    for index_center in range(len(sentence)):
        center_word = sentence[index_center]
        center_word_id = word2id[center_word]

        # left window
        index_left = max(index_center-WINDOW_SIZE,0)
        while index_left < index_center:
            left_word = sentence[index_left]
            try : #TRY TO RETRIEVE LEFT WORD ID
                left_word_id = word2id[left_word]
                list_attr = []
                list_attr.append(center_word_id)
                list_attr.append(left_word_id)
                str_att = ' '.join(str(e)+':1' for e in sorted(list_attr))
                if str_att not in cooccur_map:
                    cooccur_map[str_att] = 0
                distance = index_center - index_left
                cooccur_map[str_att] += (1/distance)              
            except:
                print("error for word "+str(left_word))
            index_left += 1
            
        # right window
        index_right = min(index_center+WINDOW_SIZE,len(sentence)-1)
        while index_right > index_center:
            right_word = sentence[index_right]
            try : #TRY TO RETRIEVE RIGHT WORD ID
                right_word_id = word2id[right_word]
                list_attr = []
                list_attr.append(center_word_id)
                list_attr.append(right_word_id)
                str_att = ' '.join(str(e)+':1' for e in sorted(list_attr))
                if str_att not in cooccur_map:
                    cooccur_map[str_att] = 0
                distance = index_right - index_center
                cooccur_map[str_att] += (1/distance) 
            except:
                print("error for word "+str(right_word))
            index_right -= 1

# write co-occurence file
with open(COOCCURENCE_FILE,'w') as f:
    for str_att,cooccur in cooccur_map.items():
        cooccur_map[str_att] /= 2 # avoid to count the co-occurrence of each pair twice
        if cooccur_map[str_att] >= MIN_COOCCUR_VALUE:
            f.write("{} {}\n".format(cooccur_map[str_att],str_att))

# write word2id file
with open(WORD2ID_FILE,'w') as f:
    for word,ix in word2id.items():
        f.write('"{}",{},0\n'.format(word,ix))

