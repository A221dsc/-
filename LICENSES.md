# 数据与依赖来源

| 编号 | 官方入口 | 许可或来源说明 | 2026-11-10 本机状态 |
| --- | --- | --- | --- |
| D1 T-Drive | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) · [官方 OneDrive](https://1drv.ms/u/s%21AsWQUIUFkRXPgsN6OCa9OB-qziGfhA) | [用户指南及 MSR-LA](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User_guide_T-drive.pdf)。限非商业用途。原始数据和衍生数据留在本地。引用 Yuan et al. KDD 2011 和 SIGSPATIAL 2010。 | 官方下载返回 403。 |
| D2 Porto | [UCI 339，DOI 10.24432/C55W25](https://doi.org/10.24432/C55W25) · [官方 ZIP](https://archive.ics.uci.edu/static/public/339/taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015.zip) | CC BY 4.0，署名 Moreira-Matias 等及 UCI。 | 首次传输收到 162,408,536 字节后超时。续传因服务器不支持字节范围失败。用户接手下载。 |
| D3 Moving AI | [2D Pathfinding Benchmarks](https://www.movingai.com/benchmarks/grids.html) | 依据选定地图的上游条款处理分发。引用 Sturtevant 2012。 | 已登记入口，等待地图实验。 |
| D4 OSM | [Geofabrik](https://download.geofabrik.de/) | ODbL 1.0，© OpenStreetMap contributors，记录快照日期和 SHA-256。 | 已登记入口，等待规模实验。 |

C++ 使用 Boost.Geometry 和 libspatialindex C API。运行使用本机已安装依赖，未向工程复制库。分别见 [Boost.Geometry](https://www.boost.org/doc/libs/1_82_0/libs/geometry/doc/html/geometry/spatial_indexes/introduction.html) 和 [libspatialindex](https://libspatialindex.org/en/latest/overview.html)。
