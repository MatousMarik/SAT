# Unequality constraints to propositional logic
##### Matouš Mařík
    Suppose we are given a CSP over some unified domain 𝐷 ∶ [𝑑1 ... 𝑑2] where all constraints are of the form 𝑣𝑖 ≤ 𝑣𝑗, 𝑣𝑖 − 𝑣𝑗 ≤ 𝑐 or 𝑣𝑖 = 𝑣𝑗 + 𝑐 for some constant 𝑐. For example
        ((𝑣2 ≤ 𝑣3) ∨ (𝑣4 ≤ 𝑣1)) ∧ (𝑣2 = 𝑣1 + 4) ∧ 𝑣4 = 𝑣3 + 3
    for 𝑣1, 𝑣2, 𝑣3, 𝑣4 ∈ [0 ... 7] is a formula belonging to this fragment. This formula is satisfied by one of two solutions: (𝑣1 ↦ 0, 𝑣2 ↦ 4, 𝑣3 ↦ 4, 𝑣4 ↦ 7), or (𝑣3 ↦ 0, 𝑣1 ↦ 3, 𝑣4 ↦ 3, 𝑣2 ↦ 7). Show a reduction of such formulas to propositional logic.

    Hint: an encoding which requires |𝑉|⋅|𝐷| propositional variables, where |𝑉| is the number of variables and |𝐷| is the size of the domain, is given by introducing a propositional variable 𝑏𝑖𝑗 for each variable 𝑣𝑖 ∈ 𝑉 and 𝑗 ∈ 𝐷, which indicates that 𝑣𝑖 ≤ 𝑗 is true.

Pozn.: Výsledná formule bude konjunkce jednotlivých omezení.

Nechť $b_{i,j}$ jsou proměnné zakódované podle nápovědy (s indexováním $V$ a $D$ od 1). Je třeba zajistit konzistenci výsledného ohodnocení přidáním podmínky:
  $$\forall i \in 1..|V|: \bigwedge_{\substack{d_1, d_2 \in D \\ d_1 \le d_2}} (\neg b_{i,d_1} \lor b_{i,d_2})$$
  - díky této podmínce se v kódování nemůže stát, že by hodnota proměnné nebyla menší, nebo rovna nějakému $j$ a zároveň by hodnota proměnné byla menší, nebo rovna nějakému $k$, kde $j \le k$, tedy zajišťuje tranzitivitu $\le$.


## Kódování podmínek
Nechť je doména $D$ uspořádaná podle "$\le$", tedy
$$\forall i,j \in 1..|D|, i \le j: d_i \le d_j$$
jinak pracujeme s $D'$, která obsahuje všechny prvky $D$ a navíc zachovává uspořádání.
### Podmínka $v_i \le v_j$
Pro každou možnou hodnotu proměnné $v_i$ je vytvořena podmínka na hodnotu proměnné $v_j$, tak aby splňovala "$\le$":
  $$\bigwedge_{d \in D} ( \neg b_{j,d} \lor b_{i,d})$$
  - to odpovídá $\forall d \in D: v_j \le d \implies v_i \le d$

### Podmínka $v_i - v_j \le c$
- podmínka je nejdříve převedena do tvaru $v_i \le v_j + c$ (plus definováno, díky výskytu další podmínky)
  $$\forall d_1 \in D: \bigwedge_{\substack{d_2 \in D \\ d_2 + c \le d_1}} (\neg b_{j,d_2} \lor b_{i, d_1})$$

### Podmínka $v_i = v_j + c$
$$\forall d_1 \in D: \bigwedge_{\substack{d_2 \in D \\ d_1 \le d_2 + c \\ d_2 + c \le d_1}} (\neg b_{j,d_2} \lor b_{i, d_1})$$


Pozn.: pokud neexistuje žádné $d_2$ splňující podmínku, pak je přidána prázdná klauzule, která značí spor.

Pozn2.: výsledná formule by šla zkrátit pouze na omezování "sousedních" hodnot $d_1, d_2$, kde $d_2$ by byly nejmenší prvky z množiny $\lbrace d|d_1 < d \rbrace$.

# Bonus
    In the previous example, try to describe an encoding with only a 𝑂(log2 |𝐷|) propositional variable for each variable 𝑣𝑖.

Celá myšlenka je založena na indexování domény pomocí binární reprezentace indexu.

Každé hodnotě z $D$ je přiřazen unikátní index (počínající od 0). Výsledná formule je potom vytvořena stejně jako v předchozím případě, akorát každá proměnná $b_{i,j}$ je nahrazena klauzulí s $\lceil \log_2 |D| \rceil$ proměnnými tak, že pro hodnotu $j$ je vybrán odpovídající binární zápis $B$, který lze reprezentovat jako $b_{\lceil \log_2 |D|\rceil }, b_{\lceil \log_2 |D| \rceil -1},...,b_1,b_0$, kde $b_i \in \lbrace 0,1\rbrace$, a výsledná kombinace pro $b_{i,j}$ z předchozího řešení vypadá takto: 
$$
\bigwedge_{k \in 0..\lceil \log_2 |D|) \rceil} \text{bin\_repr}(i,j,k)
$$
kde $\text{bin\_repr}(i,j,k)$ je buď "$\neg b_{i,k}$" pokud $b_k$ v binárním zápisu $j$ je $0$, nebo "$b_{i,k}$", pokud je totéž rovno $1$. 
- Například pro 3. hodnotu z domény $D$ - $d_3$(počítáno od 1). Je odpovídající index 2, jehož zápis je 0..010, tedy výsledná klauzule odpovídající proměnné $b_{i,d_3}$ z předchozího řešení bude $(\ldots \land \neg b_{i,2} \land b_{i,1} \land \neg b_{i,0})$.

Zároveň je třeba zakázat všechny kombinace které neodpovídají validním indexům.
$$\forall v_i \in V, \forall d \in |D|...\lceil \log_2 |D|\rceil: \neg \bigwedge_{k \in 0..\lceil \log_2 |D|) \rceil} \text{bin\_repr}(i,d,k)$$















<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>