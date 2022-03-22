# DP-elimination
##### Matouš Mařík
    One of the preprocessing steps can be to eliminate some of the variables using so-called DP-elimination (or DP-resolution). In particular, assume we have a CNF φ and a variable x which we want to eliminate. Denote
        φ0 = {C ∈ φ | ¬x ∈ C}
        φ1 = {C ∈ φ | x ∈ C}
        φr = {C ∈ φ | C ∩ {x, ¬x} = ∅}
    Namely, φ0 consists of the clauses containing negative literal ¬x, φ1 consists of the clauses containing positive literal x, φr contains the rest of the clauses. Let us now define 
        φdp = {Res(C0, C1) | C0 ∈ φ0, C1 ∈ φ1}
    where Res(C0, C1) denotes the clauses originating from C0 and C1 by resolution. Show that φ is equisatisfiable with φ′ = φr ∧ φdp.

## Splnitelná $\varphi \implies$ splnitelná $\varphi'$
- nechť $\alpha$ je nějaké úplné ohodnocení $\varphi$, které jí splňuje - tedy je jejím (úplným modelem) ... $\alpha \models \varphi$
- potom určitě $\alpha \models \varphi_r$, neboť jsou to původní klauzule z $\varphi$
- stačí ukázat, že i $\alpha \models \varphi_{dp}$
  - každou klauzuli $C_r \in \varphi_{dp}$ lze přímo odvodit rezolucí z $\varphi$, tedy $\varphi \vdash C_r$
  - každý model formule je modelem i rezolucí odvozených klauzulí, tedy $\varphi \models C_r$
  - z toho plyne že $\alpha \models \varphi_{dp}$

## Splnitelná $\varphi' \implies$ splnitelná $\varphi$
- jsou splnitelné $\varphi_r \land \varphi_{dp}$, je třeba dokázat, že z toho vyplývá, že jsou splnitelné $\varphi_0 \land \varphi_1$
- $\alpha'$ je úplný model splňující $\varphi'$, který neobsahuje proměnnou $x$
  - triviálně $\alpha' \models \varphi_r$
- nechť $C_n$ je jakákoliv z klauzulí $C_0 \backslash \lbrace \neg x \rbrace$ (kde $C_0 \in \varphi_0$ ze zadání), nebo z klauzulí $C_1 \backslash \lbrace x \rbrace$ (kde $C_1 \in \varphi_1$ ze zadání), která <ins>není</ins> modelem $\alpha'$ splněna
  - pokud takovou klauzuli nelze najít, pak $\alpha' \models \varphi_0 \land \varphi_1$
  - pro zbytek bodů se BÚNO předpokládá, že $C_n \equiv C_{n,0} \backslash \lbrace \neg x \rbrace$, kde $C_{n,0}$ je nějaká konkrétní $C_0 \in \varphi_0$
---
- protože $\alpha' \models \varphi' \implies \alpha' \models \varphi_{dp}$, pak $\text{Res}(C_{n,0}, C_1)$ jsou modelem splněny pro všechny $C_1 \in \varphi_1$
- protože $\alpha' \not \models C_n$ (protože tak byla vybrána $C_n$), pak musí platit $\alpha' \models C_1 \backslash \lbrace x \rbrace$ a to pro všechny $C_1 \in \varphi_1$, z čehož vyplývá, že platí $\alpha' \models \varphi_1$
- model $\alpha$ který vznikne rozšířením modelu $\alpha'$ tak, že $\alpha(x)=0$ splňuje $\varphi$  
  - protože model $\alpha$ obsahující $\neg x$ splní všechny klauzule z $\varphi_0$
  - díky CNF tvaru formule rozšířený model určitě splňuje všechny klauzule, které splňoval původní model


<!--
TRASH NEFUNGUJE 

## Nesplnitelná $\varphi \implies$ nesplnitelná $\varphi'$
- pokud je $\varphi$ nesplnitelná, pak to znamená, že pomocí rezoluce dokážeme z jejích klauzulí odvodit prázdnou klauzuli. Je třeba dokázat, že pokud to jde, pak je totéž možné i pro $\varphi'$.
- rezolventi podle proměnné $x$ se dají vždy definovat jako $$R = \text{Res}(C, D) \implies R = (C \backslash \lbrace x, \neg x \rbrace ) \lor (D \backslash \lbrace x, \neg x \rbrace )$$
- důležité pozorování je, že pokud rezulujeme klauzuli podle nějaké proměnné, pak se všechny její literály, kromě literálu té proměnné, musí vyskytovat ve výsledné klauzuli 
- pokud tedy posloupností rezolucí získáme spor, pak jsme museli provést nějakou posloupnost rezolucí dohromady přes všechny literály všech zúčastněných klauzulí  
##
- pokud dokážeme odvodit spor pouze pomocí klauzulí v $\varphi_r$, pak se nám to určitě povede v obou formulích
- pokud jsme schopni odvodit spor pomocí posloupnosti rezolucí, z nichž některé jsou rezoluce s rodičovskou klauzulí z BÚNO $\varphi_0$, pak se v posloupnosti pro každou takovou rezoluci musí vyskytovat i rezoluce s rodičovskou klauzulí z $\varphi_1$
  - protože jinak by výsledná klauzule obsahovala literál $\neg x$
- pro každý takový možný pár rezolucí máme odpovídající klauzuli $\in \varphi_{dp}$, se kterou můžeme provést rezoluci, v posloupnosti místo rezoluce s první klauzulí z páru a druhou rezoluci (která nemusí nutně přímo následovat, ale v celkovém grafu stromové rezoluce se musí vyskytovat ve stejné cestě ke kořenu) úplně vynechat
  - tato "náhradní" klauzule obsahuje sjednocení všech literálů z nahrazovaných klauzulí
  - Může se třeba stát, že budou v cestě dvě rezoluce přes $x$ s klauzulí z $\varphi_0$ a pouze jedna s klauzulí z $\varphi_1$, to odpovídá dvěma párům a nevadí, že klauzuli z $\varphi_1$ použijeme jakoby dvakrát, neboť se literály sjednocují
- kdybychom nakreslili graf takové rezoluce formule $\varphi'$, tak v místě vynechané rezoluce budeme mít stejnou klauzuli, jakou bychom měli v grafu pro formuli $\varphi$


<!--
přidá nebo odebere, kdo odebere x???
<!--
## Splnitelná $\varphi'\implies$ splnitelná $\varphi$
- nechť $\alpha'$ je nějaké úplné ohodnocení $\varphi'$, které jí splňuje - tedy je jejím (úplným modelem) ... $\alpha' \models \varphi'$
- potom určitě $\alpha' \models \varphi_r$, neboť jsou to původní klauzule z $\varphi'$
- stačí tedy ukázat, že $\alpha \models \varphi_0$ a $\alpha \models \varphi_1$
  - každou klauzuli $C_r \in\varphi_{dp}$ můžeme zapsat jako $C_{r,0} \vee C_{r,1}$, kde tyto klauzule jsou po řadě $C_0\backslash\{\neg x\}$ a $C_1\backslash\{x\}$, tedy konjunkce původních klauzulí bez literálů proměnné x
  - potřebujeme dokázat, že $(\alpha' \models C_r) \implies (\alpha' \models (C_0 \land C_1))$

### $(\alpha' \models C_r) \implies (\alpha' \models (C_0 \land C_1))$
- 



<!--
    ## Nesplnitelná $\varphi \implies$ nesplnitelná $\varphi'$
    - pokud je FI nesplnitelná, pak to znamená, že pomocí rezoluce dokážeme z jejích klauzulí odvodit prázdnou klauzuli. Potřebuji dokázat, že totéž je možné i pro FI'.
    - Problém, který by teoreticky mohl nastat: Z klauzulí C FI0/FI1 a D z FI odvodíme rezolucí něco, co pomocí klauzulí FI' nedokážeme odvodit. 

-->




<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>