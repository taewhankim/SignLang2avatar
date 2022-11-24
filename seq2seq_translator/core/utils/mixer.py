import click 


# helpers 
def post_process(pos_token_list):
    for pos_idx in range(len(pos_token_list)):
        if len(pos_token_list[pos_idx]) > 1:
            pos_token_list[pos_idx] = [[i[0] for i in pos_token_list[pos_idx]]]
    pos_token_list = [i[0] for i in pos_token_list]
    return pos_token_list

def get_common_elem(text1, text2):
    text1_list = list(text1)
    text2_list = list(text2)
    result_str = ''
    for str in text1_list:
        if str in text2_list:
            result_str += str
    return result_str

def get_original_sentence(text_list):
    result_str = ''
    for text in text_list:
        result_str += text 
    result_str = result_str.replace('▁',' ')
    return result_str

def batch_padding(batch_list, batch_word_token, pos_tagger):
    len_list = [len(i) for i in batch_word_token]
    max_len = max(len_list)
    for batch_idx in range(len(batch_list)):
        if len(batch_list[batch_idx]) < max_len:
            batch_list[batch_idx] = batch_list[batch_idx] + [[pos_tagger.pos_to_idx['PAD']]]*(max_len - len(batch_list[batch_idx]))
    return batch_list


def get_one_modified_pos_token(tokenized_pos, tokenized_word, pos_idx = 0, start_idx=0, special_idx = None):
    '''
    설명:
        해당 함수는 PoS(part of speach) toeknized된 것을 word tokenized 된 것을 기준으로 매칭시키는데 사용되는 함수입니다. 

        pos_idx = 매칭시키기 원하는 pos_tokenized의 단어 인덱스 
        start_idx = word_tokenized에서 관심있는 index의 번호 
    
    예:
        문장 - 하나의 기둥 위에 지어진 작은 사원입니다. 
        tokenized_word - [하나] [의]  [기] [둥] [위] [에] [지] [어] [진] [작은] [사] [원] [입니다] 
        tokenized_pos  - [하나] [의]  [기둥]  [위] [에] [지] [어] [진] [작] [은] [사원] [입니다]
        
        경우1) pos_idx = 0, start_idx = 0 인 경우 
            - pos_tokenized 에서 pos_idx = 0 인 경우는 [하나] 이므로, 이것과 매칭되는 번호를 찾는다
            - start_idx = 0 이므로, tokeninzed_word에서 idx 0부터 search를 진행한다. 
            - idx 0 에서 word tokenized가 [하나] 이므로, 동일하고 따라서 결과로 result_idx = 0 을 내보낸다. 
            - 그리고 다음 번호에 관한 정보를 주기 위해서 pos_idx 는 1을, start_idx는 1을 리턴한다. 

        경우2) pos_idx = 2, start_idx = 2 경우 
            - pos_tokenized 에서 pos_idx = 2 인 경우는 [기둥] 이므로, 이것과 매칭되는 번호를 찾는다. 
            - start_idx = 2 이므로, tokenized_word에서 idx 2부터 search를 진행한다. 
            - idx = 2에서 perfect match가 되지 않으므로, pos_tokniezd의 [기둥] 을 split 한다. 
                - split 결과로 얻어지는 것은 [기, 둥] 이다. 
                - 이제 [기] 를 start_idx = 2 에서 찾는다. 
                    - perfect match가 성립된다. 
                    - 따라서 result_idx = 2를 내보낸다. 
                    - 그리고 다음 번호에 관한 정보를 주기 위해서 pos idx는 동일하게 2를, start_idx는 3을 return 한다. 
                    - 그리고 pos_word 전부를 아직 사용하지 않았으므로, speical_idx 는 1을 주어서 [기둥] 에서 [둥] 부터 시작하도록 한다. 

    return 
        - pos_idx, start_idx, special_idx, result_idx
    '''
    tokenized_original_word = tokenized_word
    tokenized_word = [i.replace('▁','') for i in tokenized_word]
    # print(f"tokenized_pos:{tokenized_pos}")
    # print(f"tokenized_word:{tokenized_word}")
    one_pos = tokenized_pos[pos_idx][0]
    # print(f"one_pos:{one_pos}")
    if tokenized_word[start_idx] == '':
        return pos_idx, start_idx + 1, 'SPACE'

    if one_pos == tokenized_word[start_idx] and special_idx is None:
        return pos_idx + 1, start_idx + 1, None
    else:
        if one_pos in tokenized_word[start_idx]:
            if one_pos == tokenized_word[start_idx][-len(one_pos):]:
                if tokenized_word[start_idx].count(one_pos) == 1:
                    return pos_idx + 1, start_idx + 1, None
                else:
                    if pos_idx != len(tokenized_pos) - 1:
                        if tokenized_pos[pos_idx + 1][0] in tokenized_word[start_idx]:
                            return pos_idx + 1, start_idx, None
                    return pos_idx + 1, start_idx + 1, None
            else:
                return pos_idx + 1, start_idx, None

        elif tokenized_word[start_idx] in one_pos:
            if tokenized_word[start_idx] == one_pos[-len(tokenized_word[start_idx]):]:
                if one_pos.count(tokenized_word[start_idx]) == 1:
                    return pos_idx +1, start_idx+1, None 
                else:
                    if start_idx != len(tokenized_word) - 1:
                        if tokenized_word[start_idx + 1] in one_pos and tokenized_word[start_idx +1] != '':
                            return pos_idx, start_idx + 1, None
                    return pos_idx +1, start_idx+1, None 
            else:
                return pos_idx, start_idx + 1, None
        else:
            original_sentence = get_original_sentence(tokenized_original_word)
            word_start = original_sentence.find(tokenized_word[start_idx])
            pos_start = original_sentence.find(one_pos)
            if word_start > pos_start:
                return pos_idx+1, start_idx, None
            else:
                return pos_idx, start_idx+1, None                    


def modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger):
    '''
    설명:
        해당 함수는 PoS(part of speach) toeknized된 것을 word tokenized 된 것을 기준으로 매칭시키는데 사용되는 함수입니다. 
        
    
    예:
        문장 - 하나의 기둥 위에 지어진 작은 사원입니다. 
        word tokenied - [하나] [의]  [기] [둥] [위] [에] [지] [어] [진] [작은] [사] [원] [입니다] 
        pos tokenized - [하나] [의]  [기둥]  [위] [에] [지] [어] [진] [작] [은] [사원] [입니다]
        matched_pos   - [하나] [의]  [기둥] [기둥] [위] [에] [지] [어] [진] [작,은] [사원] [사원] [입니다]

        매칭방식 
            - 1) word tokenized를 기준으로 개수를 맞춘다. 
            - 2) 위 예를 참고하며, 위 예에서 [작 + 은] 은 [작] 과 [은] 을 embedding하여 합하고, 반으로 나눈것을 의미한다. 

    return 
        - pos_idx, start_idx, special_idx, result_idx
    '''
    modified_pos_token_list = []
    modified_pos_token_list.append([[pos_tagger.pos_to_idx['START']]])
    #initial setting 
    pos_idx, start_idx, special_idx, result_idx = 0, 0, None, 0 

    while pos_idx < len(tokenized_pos):
        one_idx_list = []
        while True:
            new_pos_idx, new_start_idx, special_idx = get_one_modified_pos_token(tokenized_pos, tokenized_word, pos_idx, start_idx, special_idx)
            if special_idx == 'SPACE':
                one_idx_list.append([pos_tagger.pos_to_idx['SPACE']])
            else:
                one_idx_list.append(pos_token[pos_idx])
            pos_idx = new_pos_idx
            if start_idx != new_start_idx:
                start_idx = new_start_idx
                modified_pos_token_list.append(one_idx_list)
                break
            else:
                pass
    modified_pos_token_list.append([[pos_tagger.pos_to_idx['END']]])
    modified_pos_token_list = post_process(modified_pos_token_list)
    return modified_pos_token_list

def batch_modified_pos_token(batch_tokenized_pos, batch_tokenized_word, batch_pos_token, batch_word_token, pos_tagger):
    batch_modified_pos_list = []
    for batch_idx in range(len(batch_tokenized_pos)):
        modified_pos_list = modified_pos_token(tokenized_pos= batch_tokenized_pos[batch_idx], 
                                                tokenized_word= batch_tokenized_word[batch_idx], 
                                                pos_token= batch_pos_token[batch_idx], 
                                                word_token= batch_word_token[batch_idx], 
                                                pos_tagger = pos_tagger)
        batch_modified_pos_list.append(modified_pos_list)
    batch_modified_pos_list = batch_padding(batch_modified_pos_list, batch_word_token, pos_tagger)
    return batch_modified_pos_list


# unit test 
def main(config_file = "/root/sdhan/working_code/sign_translation/dev_sd/configs/lstm_mix_embedding.yaml"):
    
    from __init__ import load_yaml
    from taggers import PosTagger

    config_file = load_yaml(config_file)
    pos_tagger = PosTagger(config_file.tagger.pos)
    # this main is for unit test 
    # case 1 
    tokenized_word = [['▁하나', '의', '▁기', '둥', '▁위', '에', '▁지', '어', '진', '▁작은', '▁사', '원', '입니다'], ['▁그는', '▁보기', '와', '는', '▁달리', '▁바', '보', '가', '▁아니다']]
    word_token = [[   2, 4928, 7095, 1258, 5913, 3552, 6896, 4297, 6855, 7344, 3939, 2573,
            7020, 7139,    3,    1,    1,    1,    1,    1,    1,    1,    1], 
            [   2, 1191, 2362, 6983, 5760, 1601, 2186, 6364, 5330, 3100,    3,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1]]
    tokenized_pos = [[('하나', 'NR'), ('의', 'JKG'), ('기둥', 'NNG'), ('위', 'NNG'), ('에', 'JKB'), ('지', 'VV'), ('어', 'EC'), ('진', 'VX+ETM'), ('작', 'VA'), ('은', 'ETM'), ('사원', 'NNG'), ('입니다', 'VCP+EC')], 
    [('그', 'NP'), ('는', 'JX'), ('보', 'VV'), ('기', 'ETN'), ('와', 'JKB'), ('는', 'JX'), ('달리', 'MAG'), ('바보', 'NNG'), ('가', 'JKC'), ('아니', 'VCN'), ('다', 'EC')]]
    pos_token = [[[23], [9], [20], [20], [7], [36], [0], [37, 3], [33], [3], [20], [35, 0]], 
                [[22], [14], [36], [4], [7], [14], [15], [20], [8], [34], [0]]]

    result = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)
    
    # case 2 
    tokenized_word = [['▁한', '▁부부', '의', '▁아름', '답', '고', '▁', '숭', '고', '한', '▁사랑을', '▁기념', '하고', '자', '▁지', '어', '졌다', '고', '▁', '하니', '▁더욱', '▁감성', '적인', '▁모습이', '네요', '.']]
    word_token = [[   2, 4955, 2436, 7095, 3109, 5801, 5439,  517, 6649, 5439, 7828, 2591,
        1263, 7788, 7147, 4297, 6855, 7250, 5439,  517, 7797, 1700,  795, 7206,
        2059, 5703,   54,    3]]
    tokenized_pos = [[('한', 'MM'), ('부부', 'NNG'), ('의', 'JKG'), ('아름답', 'VA'), ('고', 'EC'), ('숭고', 'XR'), ('한', 'XSA+ETM'), ('사랑', 'NNG'), ('을', 'JKO'), ('기념', 'NNG'), ('하', 'XSV'), ('고자', 'EC'), ('지', 'VX'), ('어', 'EC'), ('졌', 'VX+EP'), ('다고', 'EC'), ('하', 'VV'), ('니', 'EC'), ('더욱', 'MAG'), ('감성', 'NNG'), ('적', 'XSN'), ('인', 'VCP+ETM'), ('모습', 'NNG'), ('이', 'VCP'), ('네요', 'EF'), ('.', 'SF')]]
    pos_token = [[[17], [20], [9], [33], [0], [39], [40, 3], [20], [10], [20], [42], [0], [37], [0], [37, 2], [0], [36], [0], [15], [20], [41], [35, 3], [20], [35], [1], [26]]]

    result2 = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)

    # case 3 
    # tokenized word - 일주일 
    # tokenized pos  - 일 / 주일 
    tokenized_word = [['일주일', '동안', '다른', '도시', '에', '', '가', '있어', '요']]
    word_token = [[   2, 4955, 2436, 7095, 3109, 5801, 5439,  517, 6649, 5439, 7828, 2591,
        1263, 7788, 7147, 4297, 6855, 7250, 5439,  517, 7797, 1700,  795, 7206,
        2059, 5703,   54,    3]]
    tokenized_pos = [[('일', 'NR'), ('주일', 'NNBC'), ('동안', 'NNG'), ('다른', 'MM'), ('도시', 'NNG'), ('에', 'JKB'), ('가', 'VV+EC'), ('있', 'VX'), ('어요', 'EF')]]
    pos_token = [[[17], [20], [9], [33], [0], [39], [40, 3], [20], [10], [20], [42], [0], [37], [0], [37, 2], [0], [36], [0], [15], [20], [41], [35, 3], [20], [35], [1], [26]]]

    result3 = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)

    # case 4 
    # tokenized word - 그가 / 그 
    # tokenized pos  - 그 / 가 / 그 
    tokenized_word = [['그가', '그', '소설', '을', '쓴', '지', '50', '년', '이', '지', '났다']]
    word_token = [[   2, 4955, 2436, 7095, 3109, 5801, 5439,  517, 6649, 5439, 7828, 2591,
        1263, 7788, 7147, 4297, 6855, 7250, 5439,  517, 7797, 1700,  795, 7206,
        2059, 5703,   54,    3]]
    tokenized_pos = [[('그', 'NP'), ('가', 'JKS'), ('그', 'MM'), ('소설', 'NNG'), ('을', 'JKO'), ('쓴', 'VV+ETM'), ('지', 'NNB'), ('50', 'SN'), ('년', 'NNBC'), ('이', 'JKS'), ('지났', 'VV+EP'), ('다', 'EC')]]
    pos_token = [[[17], [20], [9], [33], [0], [39], [40, 3], [20], [10], [20], [42], [0], [37], [0], [37, 2], [0], [36], [0], [15], [20], [41], [35, 3], [20], [35], [1], [26]]]
    result4 = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)

    # case 5 
    # tokenized word - 이민 / 자들이
    # tokenized pos  - 이민자 / 들 / 이 
    tokenized_word = [['싱가포르', '차', '이나', '', '타운', '은', '중국', '이민', '자들이', '모여', '살', '면서', '조성', '되었', '습니다']]
    word_token = [[   2, 4955, 2436, 7095, 3109, 5801, 5439,  517, 6649, 5439, 7828, 2591,
        1263, 7788, 7147, 4297, 6855, 7250, 5439,  517, 7797, 1700,  795, 7206,
        2059, 5703,   54,    3]]
    tokenized_pos = [[('싱가포르', 'NNP'), ('차이', 'NNG'), ('나', 'JC'), ('타운', 'NNG'), ('은', 'JX'), ('중국', 'NNP'), ('이민자', 'NNG'), ('들', 'XSN'), ('이', 'JKS'), ('모여', 'VV+EC'), ('살', 'VV'), ('면서', 'EC'), ('조성', 'NNG'), ('되', 'XSV'), ('었', 'EP'), ('습니다', 'EF')]]
    pos_token = [[[17], [20], [9], [33], [0], [39], [40, 3], [20], [10], [20], [42], [0], [37], [0], [37, 2], [0], [36], [0], [15], [20], [41], [35, 3], [20], [35], [1], [26]]]
    result5 = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)
    

    # case 6 
    tokenized_word = [['제주도', '는', '▁다른', '▁지역', '▁보다', '', '박물관', '수', '가', '▁평균', '6', '배', '가', '넘는', '다']]
    word_token = [[   2, 4955, 2436, 7095, 3109, 5801, 5439,  517, 6649, 5439, 7828, 2591,
        1263, 7788, 7147, 4297, 6855, 7250, 5439,  517, 7797, 1700,  795, 7206,
        2059, 5703,   54,    3]]
    tokenized_pos = [[('제주도', 'NNP'), ('는', 'JX'), ('다른', 'MM'), ('지역', 'NNG'), ('보다', 'JKB'), ('박물관', 'NNG'), ('수', 'NNG'), ('가', 'JKS'), ('평균', 'NNG'), ('6', 'SN'), ('배', 'NNG'), ('가', 'JKS'), ('넘', 'VV'), ('는다', 'EC')]]
    pos_token = [[[17], [20], [9], [33], [0], [39], [40, 3], [20], [10], [20], [42], [0], [37], [0], [37, 2], [0], [36], [0], [15], [20], [41], [35, 3], [20], [35], [1], [26]]]
    result6 = batch_modified_pos_token(tokenized_pos, tokenized_word, pos_token, word_token, pos_tagger = pos_tagger)
    print('unit test clear')


if __name__ == "__main__":
    main()
