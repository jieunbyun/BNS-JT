---
layout: post
title: MBNPy's system catalogue
usemathjax: true
tags: [system-type, functionality]
categories: user-guide
---

<p>There is no free lunch and efficiency of MBNPy is no exception. While simulation-based methods like MCS are applicable to any general systems, MBNPy requires system-specific algorithms to quantify a system's probability distribution.</p>

<p>We note that for all systems below, conventional approach has complexity of $O(M^N)$. Usually, with a 16GB RAM, many software will raise a warning for exceeding $2^{33}$; this means 33 binary-state components.</p>

<div style="text-align: center;">
    <p><strong>Table 1: MBNPy's system catalogue.</strong></p>
    <table>
        <thead>
            <tr>
                <th>System type<sup>1</sup></th>
                <th>Complexity<sup>2</sup></th>
                <th>Application examples</th>
                <th>Ref.</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Series/parallel system (B)</td>
                <td>$O(N)$</td>
                <td>Structural systems</td>
                <td>[1]</td>
            </tr>
            <tr>
                <td>Series-parallel system (B/M)</td>
                <td>Worst case: $O(NM)$</td>
                <td>Mechanical systems</td>
                <td>[2]</td>
            </tr>
            <tr>
                <td>$k$-out-of-$N$ sysem (B/M)</td>
                <td> To be computed. </td>
                <td>Oil distribution system</td>
                <td>[2]</td>
            </tr>
            <tr>
                <td>Coherent system (B/M)<sup>3</sup></td>
                <td>$O(\mathcal{R})$</td>
                <td>Maximum flow, shortest path</td>
                <td>[3]</td>
            </tr>
        </tbody>
    </table>
</div>
<p>
  <small>
    <strong>1.</strong> B and M denote binary- and multi-state respectively. <br>
    <strong>2.</strong> $N$ refers to the number of components, which is the primary bottleneck. $M$ is the number of states. $\mathcal{R}$ denotes the number of failure and survival mechanisms. <br>
    <strong>3.</strong> This definition is very broad, covering all systems above.<br>
    <strong>References:</strong> <br>
    <strong>[1]</strong> Byun, J. E., Zwirglmaier, K., Straub, D. & Song, J. (2019). <a href="https://doi.org/10.1016/j.ress.2019.01.007">Matrix-based Bayesian Network for efficient memory storage and flexible inference.</a> <em>Reliability Engineering & System Safety</em>, 185, 533-545. <br>
    <strong>[2]</strong> Byun, J. E. & Song, J. (2021). <a href="https://doi.org/10.1016/j.ress.2021.107468">Generalized matrix-based Bayesian network for multi-state systems.</a> <em>Reliability Engineering & System Safety</em>, 211, 107468. <br>
    <strong>[3]</strong> Byun, J. E., Ryu, H. & Straub, D. (in preparation). Branch and bound algorithm for efficient reliability analysis of general coherent systems.
  </small>
</p>
