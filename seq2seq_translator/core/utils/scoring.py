from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import Levenshtein as Lev
from typing import List
import os
import re

smoothing = SmoothingFunction().method1


def get_score(tokenizer, reference: List, hyphothesis: List):

    bleu_score = get_bleu_score(tokenizer, reference, hyphothesis)
    return {"bleu_score": bleu_score}


def get_bleu_score(tokenizer, tokenizer_name, reference, hyphothesis):
    bleu_score = 0
    smoothing = SmoothingFunction().method1
    for idx in range(len(reference)):
        if tokenizer_name == "sentencepiece":
            reference_sentence = tokenizer.IdToPiece(reference[idx])
            hyphothesis_sentence = tokenizer.IdToPiece(hyphothesis[idx])
        elif tokenizer_name == "KoBertVocab":
            # remove cls and sep token
            reference_sentence = tokenizer.tokenizer.convert_ids_to_tokens(
                reference[idx]
            )[1:-1]
            if len(hyphothesis.size()) == 1:
                hyphothesis_sentence = tokenizer.tokenizer.convert_ids_to_tokens(
                    hyphothesis
                )[1:-1]
            else:
                hyphothesis_sentence = tokenizer.tokenizer.convert_ids_to_tokens(
                    hyphothesis[idx]
                )[1:-1]

        bleu_score += sentence_bleu(
            [reference_sentence],
            hyphothesis_sentence,
            [0.5, 0.5, 0, 0],
            smoothing_function=smoothing,
        )
    return bleu_score


def get_bleu_score_moses(human_reference_text, machine_pred_text):
    """
    args:
        human_reference_text - 사람이 번역한 리스트
        machine_pred_text - 기계가 번역한 리스트
        각 txt 파일은 detokenized 된 형태로 주어져야한다. (raw text, tokenized 된 형태가 아님)

        각 결과값은 line별로 bleu를 구하게 된다.
    return:
        dict of
        bleu, bleu[1], bleu[2], bleu[3], bleu[4]
    """
    bleu_dic = dict()

    bleu_file_path = os.path.abspath(__file__)
    rfind = bleu_file_path.rfind("/")
    BLEU_FILE_PATH = bleu_file_path[:rfind]
    
    if not os.path.exists(f"{BLEU_FILE_PATH}/result"):
        os.mkdir(f"{BLEU_FILE_PATH}/result")
        
    with open(f"{BLEU_FILE_PATH}/result/human_ref_text.txt", "w") as f:
        f.writelines(human_reference_text)
    with open(f"{BLEU_FILE_PATH}/result/machine_text.txt", "w") as f:
        f.writelines(machine_pred_text)

    stream = os.popen(
        f"perl {BLEU_FILE_PATH}/multi-bleu-detok.perl {BLEU_FILE_PATH}/result/human_ref_text.txt < {BLEU_FILE_PATH}/result/machine_text.txt"
    )
    output = stream.read()

    bleu_point = output.find(",")
    end_point = output.find("(")

    split_point = [m.start() for m in re.finditer("/", output)]

    bleu = float(output[:bleu_point][6:])
    bleu1 = float(output[bleu_point + 1 : split_point[0]])
    bleu2 = float(output[split_point[0] + 1 : split_point[1]])
    bleu3 = float(output[split_point[1] + 1 : split_point[2]])
    bleu4 = float(output[split_point[2] + 1 : end_point])

    bleu_dic["bleu"] = bleu
    bleu_dic["bleu1"] = bleu1
    bleu_dic["bleu2"] = bleu2
    bleu_dic["bleu3"] = bleu3
    bleu_dic["bleu4"] = bleu4

    return bleu_dic


def get_sentence_bleu(hyphothesis_sentence, reference_sentence):
    bleu_score = sentence_bleu(
        [reference_sentence],
        hyphothesis_sentence,
        [0.5, 0.5, 0, 0],
        smoothing_function=smoothing,
    )
    return bleu_score


def get_wer(ref, hyp, debug=False):
    """
    ref : target sentence
    hyp : infer sentence
    """
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY = 1  # Tact
    INS_PENALTY = 1  # Tact
    SUB_PENALTY = 1  # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = (
                    costs[i - 1][j - 1] + SUB_PENALTY
                )  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
    return (numSub + numDel + numIns) / (float)(
        len(r)
    )  # numCor, numSub, numDel, numIns,


def get_cer(ref, hyp):
    """
    ref : target sentence
    hyp : infer sentence
    """
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    dist = Lev.distance(hyp, ref)
    length = len(ref)
    return dist / length  # dist, length,


# # unit test
# if __name__ == "__main__":
#     reference_sentence = [2, 4092, 6516, 1723, 2368, 2468, 7941, 5495, 3862, 3]
#     hyphothesis_sentence = [2, 4092, 6516, 1723, 2368, 2468, 7941, 5495, 3862, 3]
#     smoothing = SmoothingFunction().method1
#     bleu_score = sentence_bleu(
#         [reference_sentence],
#         hyphothesis_sentence,
#         [0.5, 0.5, 0, 0],
#         smoothing_function=smoothing,
#     )
#     print(bleu_score)
