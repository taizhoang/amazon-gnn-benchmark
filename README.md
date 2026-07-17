# Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV

<table>
  <tr>
    <td><img src="https://www.kaggleusercontent.com/kf/334701497/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..geHNd-xDo-rMCKIrBGgN2A.26OlgBM_BQ3jv0QOX-k3G5Row-mCDx9orLkHA7eT0fuaI3Ij58fJabeu47dg-HMDUi-XtCPHKn_phjABXuXfDeVRMKlkozixny19ioL3ZFSlg7Prjx4pNVdENG0dZCeqD5IZE2Ri8Jr6uXUvsgTyXQjgCvjyFGrJRp2Fj-KRB2tHmrDFY2vPoL8hoQ6oGaQp0k0AYsEpbMCVJd8uEz_BAY0_jt97JCxViYrdKow06ZkUg-9lq96C4K_rETIp1mq5JUyZ_u4aBEBiK5WbHBAdBDt89cAH5sMp7I0Brhn9QBy0NZoL4MpJJlS0jEK3x8sP2ySt-I6G8j_CtAnsnKGizdsiiigl_fu8-604xRdSbTm185ZxIFflxXxka_q9gRJ42f-CUlRLatvofeQclJc1Uv0Ues9BjyJKwnv50oP55st3rjveQ5z7WTTKX_bXbww-iYP0RtcatnhcmcmqHNKU74Lg8uF57FfdHV37k0Hs5dbCamcRoK2AMs7ipM5BtxXNrwCD4RaaIoAchyIa-myBHl66BgsGmR0_Y5SfAgzd_aK7h7dWwbKa0J9-zobl8wcyyuJ5q-XAwtL-VDBsYV3O9OY6MvoBqGaF0egCEyJ9ZFZquBuCP8MRE4As5gPv0X94.U-jpofQ2Pr6JkVK3lyjsCQ/__results___files/__results___22_0.png" width="300"></td>
    <td><img src="https://www.kaggleusercontent.com/kf/334701497/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..geHNd-xDo-rMCKIrBGgN2A.26OlgBM_BQ3jv0QOX-k3G5Row-mCDx9orLkHA7eT0fuaI3Ij58fJabeu47dg-HMDUi-XtCPHKn_phjABXuXfDeVRMKlkozixny19ioL3ZFSlg7Prjx4pNVdENG0dZCeqD5IZE2Ri8Jr6uXUvsgTyXQjgCvjyFGrJRp2Fj-KRB2tHmrDFY2vPoL8hoQ6oGaQp0k0AYsEpbMCVJd8uEz_BAY0_jt97JCxViYrdKow06ZkUg-9lq96C4K_rETIp1mq5JUyZ_u4aBEBiK5WbHBAdBDt89cAH5sMp7I0Brhn9QBy0NZoL4MpJJlS0jEK3x8sP2ySt-I6G8j_CtAnsnKGizdsiiigl_fu8-604xRdSbTm185ZxIFflxXxka_q9gRJ42f-CUlRLatvofeQclJc1Uv0Ues9BjyJKwnv50oP55st3rjveQ5z7WTTKX_bXbww-iYP0RtcatnhcmcmqHNKU74Lg8uF57FfdHV37k0Hs5dbCamcRoK2AMs7ipM5BtxXNrwCD4RaaIoAchyIa-myBHl66BgsGmR0_Y5SfAgzd_aK7h7dWwbKa0J9-zobl8wcyyuJ5q-XAwtL-VDBsYV3O9OY6MvoBqGaF0egCEyJ9ZFZquBuCP8MRE4As5gPv0X94.U-jpofQ2Pr6JkVK3lyjsCQ/__results___files/__results___29_0.png" width="300"></td>
  </tr>
</table>

*Figure: Comparison of generalizability among GCN and GraphSAGE variants.*

***

This is the code for *Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV*, a project for *IT5429E - Graph analytics for big data* (Master's course @ HUST).

> All experiments are conducted in a Kaggle notebook environment equipped with an NVIDIA Tesla T4 GPU (16GB VRAM), 13GB RAM, and 2 CPU cores, using PyTorch and PyTorch Geometric for implementation.

The mentioned Kaggle notebook can be found [[here]](https://www.kaggle.com/code/thaimeuu/it5429e-workspace), which uses Kaggle utility scripts that are based on this repository.

| URL | Description | Based on |
|---|---|---|
https://www.kaggle.com/code/thaimeuu/it5429e-workspace | Experiments (outputs, which are used in the reported, are saved in this notebook) | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-models | All models | [models/](models/) |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-learning | Training and evaluation functions | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-utils | Utils functions | [utils.py](utils.py) |

The figures for OGBN-ARXIV in the report can be found here: [images/](images/)

The figures for the experiments can be found [in this Kaggle notebook](https://www.kaggle.com/code/thaimeuu/it5429e-workspace)

The checkpoints for the experiments can be found here: [checkpoints/](checkpoints/)

| File | Description | Section in report |
|---|---|---|
| [checkpoints/transductive/](checkpoints/transductive/) | The main result | Section 4.1, Table 3 |
| [checkpoints/oversmoothing/](checkpoints/oversmoothing/) | Over-smoothing analysis | Section 4.2, Figures 5, 6, 7 |
| [checkpoints/ablation/](checkpoints/ablation/) | Ablation study | Section 4.3, Table 4 |
| [checkpoints/appendix/](checkpoints/appendix/) | Over-smoothing comparison GTCN vs GCN variants | Appendix A, Figures 8, 9 |
