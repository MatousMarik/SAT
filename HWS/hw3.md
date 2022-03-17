# DP-elimination
##### Matouš Mařík
    One of the preprocessing steps can be to eliminate some of the variables using so-called DP-elimination (or DP-resolution). In particular, assume we have a CNF φ and a variable x which we want to eliminate. Denote
        φ0 = {C ∈ φ | ¬x ∈ C}
        φ1 = {C ∈ φ | x ∈ C}
        φr = {C ∈ φ | C ∩ {x, ¬x} = ∅}
    Namely, φ0 consists of the clauses containing negative literal ¬x, φ1 consists of the clauses containing positive literal x, φr contains the rest of the clauses. Let us now define φdp = {Res(C0, C1) | C0 ∈ φ0, C1 ∈ φ1} where Res(C0, C1) denotes the clauses originating from C0 and C1 by resolution. Show that φ is equisatisfiable with φ′ = φr ∧ φdp.

## Nesplnitelná $\varphi \implies$ nesplnitelná $\varphi'$
- pokud je FI nesplnitelná, pak to znamená, že pomocí rezoluce dokážeme z jejích klauzulí odvodit prázdnou klauzuli. Potřebuji dokázat, že totéž je možné i pro FI'.
- Problém, který by teoreticky mohl nastat: Z klauzulí C FI0/FI1 a D z FI odvodíme rezolucí něco, co pomocí klauzulí FI' nedokážeme odvodit. 

## Nesplnitelná $\varphi'\implies$ nesplnitelná $\varphi$
- pokud je FI' nesplnitelná, pak to znamená, že v rezoluci dojdeme k $\bot$
- víme, že 
- 

## Splnitelná $\varphi \ \backslash \ \{C\}$ => splnitelná $\varphi$
- $l$ ... blokující literál $C$
- pokud φ obsahuje pouze jednu klauzuli $C$ pak $\varphi$ je vždy splněna a neprázdná $C$ je vždy splnitelná
- $l_D$... literál $C \ \backslash \ \{l\}$, pro který platí $\neg l_D \in D$
- pokud $\varphi \ \backslash \ \{C\}$ je splnitelná, pak existuje model (který je úplné ohodnocení všech literálů $\varphi$) $a$, takový, že:
  1. buď $a(l) = 1$ => $a \models C$ => $\varphi$ je splnitelná
  2. nebo $a(l) = 0$
     - existuje-li nějaký $l_D \in a$ pak $a \models C$
     - --
     - jinak by pro všechny $D \in \varphi \ \backslash \ \{C\}$ jejich literál $\neg l_D$ (literál vyplývající z tautologie vznikající rezolucí, definovaný výše) byl v $a$, tedy $a(l_D) = 0$
     - a tedy model $a'$ t.ž. pro každý literál $l' \neq l$ platí $a'(l') = a(l')$ a zároveň $a'(l) = 1$ (tedy model, který se od $a$ liší tím, že místo $\neg l$ obsahuje $l$), splňuje jak $C$, tak $\varphi \ \backslash \ \{C\}$
     - tedy $a'$ splňuje bod 1.






<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>