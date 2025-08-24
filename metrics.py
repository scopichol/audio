import re
import unicodedata
import jiwer 
import difflib
import glob
import os
import csv

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

def seq_match_ratio(ref: str, hyp: str) -> float:
    """
    Повертає коефіцієнт схожості (0..1) між референсом і гіпотезою за difflib.SequenceMatcher.
    """
    rn = normalize(ref)
    hn = normalize(hyp)
    return difflib.SequenceMatcher(None, rn, hn).ratio()

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
    print("SequenceMatcher ratio:", seq_match_ratio(r, h))
    # Порівняння для всіх output*.txt з каталогу out
    ref_path = "sentences/sentence_1.txt"
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    except Exception:
        ref_text = ""
    metrics_rows = []
    for hyp_file in sorted(glob.glob("out/output*.txt")):
        with open(hyp_file, "r", encoding="utf-8") as f:
            hyp_text = f.read().strip()
        wer_val = wer(ref_text, hyp_text)
        cer_val = cer(ref_text, hyp_text)
        seq_ratio = seq_match_ratio(ref_text, hyp_text)
        jiwer_wer = jiwer.wer(normalize(ref_text), normalize(hyp_text))
        jiwer_cer = jiwer.cer(normalize(ref_text), normalize(hyp_text))
        wil = jiwer.wil(normalize(ref_text), normalize(hyp_text))
        mer = jiwer.mer(normalize(ref_text), normalize(hyp_text))
        wip = jiwer.wip(normalize(ref_text), normalize(hyp_text))
        print(f"\n{os.path.basename(hyp_file)}")
        print(f"WER: {wer_val*100:.2f}%, CER: {cer_val*100:.2f}%")
        print(f"SequenceMatcher ratio: {seq_ratio:.4f}")
        print(f"jiwer WER: {jiwer_wer*100:.2f}%")
        print(f"jiwer CER: {jiwer_cer*100:.2f}%")
        print("WIL:", wil)
        print("MER:", mer)
        print("WIP:", wip)
        metrics_rows.append([
            os.path.basename(hyp_file),
            wer_val, cer_val, seq_ratio,
            jiwer_wer, jiwer_cer, wil, mer, wip,
            hyp_text  # додаємо текст гіпотези
        ])
    # Запис у CSV
    with open("out/metrics.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "file", "wer", "cer", "seq_match_ratio",
            "jiwer_wer", "jiwer_cer", "wil", "mer", "wip", "text"
        ])
        writer.writerows(metrics_rows)
