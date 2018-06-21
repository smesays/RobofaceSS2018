# instructions HAVE to be exactly in this format for the parser to work properly.
# <emotion> phrase(s) (sleep)
# anytime you want a new emotion or sleep for some seconds, you have to write all 3 components. 
instructions = '<smile> Hi. My name is G. (0) <neutral> How are you? (10) <sad> I cannot see you. (4) <smile> Ah there you are. (10)'


import re
phrases = re.split('[<>()]', instructions)
comp = 0
emotion_list, phrase_list, sleep_list = [], [], []
for i in phrases:
    i = i.strip()
    if i != '':
        comp += 1
        if comp % 3 == 1:
            emotion_list.append(i)
        elif comp % 3 == 2:
            phrase_list.append(i)
        else:
            sleep_list.append(i)
print(emotion_list)
print(phrase_list)
print(sleep_list)
