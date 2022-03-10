# Equisatisfiability of CNF with blocked clause
##### Matouš Mařík
    Let φ be a CNF and C ∈ φ a clause in φ and l ∈ C a literal in C. We say that C is blocked by literal l if for every other clause D ∈ φ which contains ¬l we have that Res(C, D) is a tautology (i.e. there is another literal l′ ∈ C such that ¬l′ ∈ D). Show that φ is equisatisfiable with φ \ {C}, i.e. φ is satisfiable if and only if φ \ {C} is satisfiable.

## Splnitelná φ => splnitelná φ \ {C}
- triviální - stejný model

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