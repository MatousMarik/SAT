# Hamiltonský cyklus
##### Matouš Mařík
    Let G = (V, E) be an undirected graph. Suggest a propositional formula that is satisfi-
    able it and only if G contains a Hamiltonian cycle.

Proměnné:
- $x_{i,j}$ ... na i-té pozici cyklu je vrchol j
  - tedy $i, j \in 1..|V|$
  
Podmínky:
1. Každý vrchol musí být v cyklu:
$$\forall j: \bigvee_i x_{i,j}$$
2. Na každé pozici cyklu musí být vrchol:
$$\forall i: \bigvee_j x_{i,j}$$
3. Žádný vrchol se nesmí v cyklu objevit dvakrát:
$$\forall i,j: \forall k > i: \neg x_{i,j} \vee \neg x_{k,j}$$
4. Na každé pozici cyklu může být pouze jeden vrchol:
$$\forall i,j: \forall k > j: \neg x_{i,j} \vee \neg x_{i,k}$$
5. Na sousedních pozicích v cyklu můžou být pouze vrcholy spojené hranou:
   - Pro přehlednost pracuji s následujícími předpoklady:
     - $(j,k)$ se v tomto případě chová jako uspořádaná dvojice, tedy pokud není definována hrana $(j,k)$, pak vzniká podmínka pro $(j,k)$ i pro $(k,j)$
     - $|V|+1 = 1$ (pro uzavření cyklu)
       - Tento problém by šel také vyřešit přidáním poslední pozice, na které by musel být stejný vrchol jako na první pozici a na kterou by se nevztahovala podmínka 3
$$\forall (j,k) \notin E: \neg x_{i,j} \vee \neg x_{i+1,k},\ i=1..|V|$$

Výsledná výroková formule vznikne konjunkcí všech těchto podmínek a jelikož každá z nich je klauzulí, tak se bude ve tvaru CNF.






<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>