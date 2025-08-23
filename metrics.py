import re
import unicodedata
import jiwer 

def normalize(text: str) -> str:
    # юнікод-нормалізація, нижній регістр
    t = unicodedata.normalize("NFKC", text).lower()
    # уніфікація апострофів
    t = t.replace("’", "'").replace("ʼ", "'").replace("`", "'")
    # прибрати все, крім букв/цифр/пропусків/апострофа
    t = re.sub(r"[^\w\s']", " ", t, flags=re.UNICODE)
    # злиплі пробіли → один
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _edit_ops(ref_tokens, hyp_tokens):
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    back = [[None]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        dp[i][0] = i; back[i][0] = 'D'
    for j in range(1, n+1):
        dp[0][j] = j; back[0][j] = 'I'
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
                back[i][j] = 'C'
            else:
                choices = [(dp[i-1][j-1]+1,'S'), (dp[i-1][j]+1,'D'), (dp[i][j-1]+1,'I')]
                dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])
    # підрахунок S/D/I/C
    i, j = m, n
    S=D=I=C=0
    while i>0 or j>0:
        op = back[i][j]
        if op == 'C': C+=1; i-=1; j-=1
        elif op == 'S': S+=1; i-=1; j-=1
        elif op == 'D': D+=1; i-=1
        elif op == 'I': I+=1; j-=1
    return S, D, I, C, m, n

def wer(ref: str, hyp: str) -> float:
    rt = normalize(ref).split()
    ht = normalize(hyp).split()
    S,D,I,C,Nref,Nhyp = _edit_ops(rt, ht)
    return (S+D+I) / max(1, Nref)

def cer(ref: str, hyp: str) -> float:
    # по символах (без пробілів, але після нормалізації)
    r = normalize(ref).replace(" ", "")
    h = normalize(hyp).replace(" ", "")
    S,D,I,C,Nref,Nhyp = _edit_ops(list(r), list(h))
    return (S+D+I) / max(1, Nref)

def ser(ref_sentences, hyp_sentences) -> float:
    # списки рядків однакової довжини; кожну пару оцінюємо як 0/1
    assert len(ref_sentences) == len(hyp_sentences)
    bad = 0
    for r, h in zip(ref_sentences, hyp_sentences):
        rt = normalize(r).split(); ht = normalize(h).split()
        S,D,I,_,Nref,_ = _edit_ops(rt, ht)
        if S+D+I > 0: bad += 1
    return bad / max(1, len(ref_sentences))

if __name__ == "__main__":
    r = "The birch canoe slid on the smooth planks."
    h = "The badge, the noise, slipped on the small supplanks."
    print("REF:", r)
    print("HYP:", h)
    print(f"WER: {wer(r,h)*100:.2f}%, CER: {cer(r,h)*100:.2f}%")

    rn = normalize(r)
    hn = normalize(h)
    # WER (Word Error Rate), CER (Character Error Rate)
    print(f"WER: {jiwer.wer(rn,hn)*100:.2f}%, CER: {jiwer.cer(rn,hn)*100:.2f}%")
    # WIL (Word Information Lost), MER (Match Error Rate), WIP (Word Information Preserved)
    print("WIL:", jiwer.wil(rn,hn))
    print("MER:", jiwer.mer(rn,hn))
    print("WIP:", jiwer.wip(rn,hn))
