import re
from konlpy.tag import Twitter
twt = Twitter()

def preproc(line, is_nl=False):
    # pre1 괄호제거
    pre1 = "[\(\[\<\]\)][ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:]+[\(\[\<\>\]\)]"
    # pre2 특문제거
    pre2 = "[^ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:]"
    # pre34 대문자, 숫자 떨어뜨리기 + 앞 뒤 스페이스
    pre3 = "([A-Z][^a-zA-Z ]|[^a-zA-Z ][A-Z])"
    pre4 = "[A-Z]{2,100}|[0-9]+"
    # pre5 = 영단어 앞, 뒤 스페이스
    pre5 = "[A-Z][a-z]+|[a-z]+|[.,!?~:]+|[0-9]+"
    # pre6 = 스페이스 2회 이상 -> 1회
    pre6 = "[ ]{2,}"

    l = line.strip().strip("\n")
    l = re.sub(pre1, "",l)
    l = re.sub(pre2, " ", l)
    uppers = re.findall(pre3,l)
    for upp in uppers: 
        re_case = " "+" ".join([u for u in upp])+" "
        l = l.replace(upp, re_case)
    uppers = re.findall(pre4,l)
    for upp in uppers: 
        re_case = " "+" ".join([u for u in upp])+" "
        l = l.replace(upp, re_case)
    words = re.findall(pre5, l)

    # 형태소 분석에 의한 split
    if is_nl:
        tags = twt.pos(l)
        l = " ".join([i for i, j in tags]) 

    for w in words:
        l = l.replace(w, " "+w+" ")

    spaces = re.findall(pre6, l)
    for s in spaces:
        l = l.replace(s, " ")

    spaces = re.findall(pre6, l)
    for s in spaces:
        l = l.replace(s, " ")    

    return l.strip()