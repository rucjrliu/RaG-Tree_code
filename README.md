# RaG-Tree_code

This is the code corresponding to the paper: RaG-Tree: Combining R-Tree and HNSW for Multi-Attribute Range Filtered Approximate Nearest Neighbor Search. Multi-attribute range-filtered approximate nearest neighbor search (MR-ANNS), which retrieves high-dimensional vectors satisfying multiple attribute constraints, is a fundamental operation in modern AI applications. Existing MR-ANNS indexes either exploit a single attribute for range localization or recursively partition objects along individual attributes, which may limit their ability to exploit attribute correlations for effective range pruning and attribute-vector correlations for efficient nearest-neighbor search. In this paper, we propose RaG-Tree, a unified index that couples an R-tree with partition-aware HNSW graphs for MR-ANNS. RaG-Tree leverages hierarchical R-tree partitions for effective range pruning and adapts the sparsity of each HNSW graph to the local vector distributions within its partition, enabling lightweight indexing and efficient query processing. To support efficient query processing and dynamic updates, we develop a cost-based adaptive search algorithm that minimizes unnecessary graph exploration, together with an efficient index maintenance mechanism for incrementally updating affected partition-aware HNSW graphs. Extensive experiments on three real-world datasets show that RaG-Tree achieves superior query performance over state-of-the-art baselines, while also providing lightweight indexing and fast incremental updates.

## Environment, Datasets and Indexfiles

**Step 1:** Download this code and unzip it.

**Step 2:** Install dependencies of the environment fanns.

    cd RaG-Tree_code/
    conda install -f environment.yml

**In the following, all running should be at this directory 'RaG-Tree_code/'.**

**Step 3:** Downloading our experimental data.

Download file **dataset_and_indexfiles.tar** (the data and indexfiles containing in this tar file is necessary for all experiments in the paper) from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/IQCgKCG24AfdRJCpiFNxvkp-AV0xltaj2qmDM7lt4s3GEKA?e=FZor9V and then decompress it:

    mkdir data
    mkdir indexfiles
    mkdir output
    tar -xf dataset_and_indexfiles.tar
    mv dataset_and_indexfiles/data/* data/
    mv dataset_and_indexfiles/indexfiles/* indexfiles/

Notice: 120GB and more storage space.

In each sub directory of a dataset name, the *name_data.npz* is the dataset, the *name_selrandom_query.npz* is the test workload, and the *name_selrandom_gt.npz* is the groundtruth of the test workload. These files can all be opened by Numpy.

## Query Search Performance

Firstly, compile test_ragtree_s.cpp

    cd test_ragtree
    ./compile_test_ragtree_s.sh

Afterwards, run the search performance experiments by

    ./run_ragtree_s.sh 1 <dataset> a <CAS> <top-k> <ef_search>

For example,

    ./run_ragtree_s.sh 1 laion a o 10 2000

for \<dataset\>=laion, **turn on** CAS, \<top-k>\=10, \<ef_search\>=2000. And

    ./run_ragtree_s.sh 1 msmarco a d 10 1000

for \<dataset\>=msmarco, **without** CAS, \<top-k>\=10, \<ef_search\>=1000.

### Recommended \<ef_search\> Setting

**dblp**:

100-4000 for \<CAS\>=d

50-2500 for \<CAS\>=o

**msmarco**:

300-20000 for \<CAS\>=d

150-10000 for \<CAS\>=o

**laion**

500-25000 for \<CAS\>=d

150-5000 for \<CAS\>=o

## Scalability with the Number of Attributes

Firstly, download file **dataset_and_indexfiles_for_varying_attrnum.tar** (the data and indexfiles containing in this tar file is necessary for all experiments in the paper) from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/IQBtg-0LNRp3TJ3GtpjvL0J6Ac6RRLVizLYeYNc6qDxzSvI?e=z6TiBv and then decompress it:

    tar -xf dataset_and_indexfiles_for_varying_attrnum.tar
    mv dataset_and_indexfiles_for_varying_attrnum/data/* data/
    mv dataset_and_indexfiles_for_varying_attrnum/indexfiles/* indexfiles/

Notice: another 260GB and more storage space.

Then,

    cd test_ragtree

Afterwards, run the search performance experiments by

    ./run_ragtree_s.sh 1 <dataset> a <CAS> <top-k> <ef_search>

For example,

    ./run_ragtree_s.sh 1 laion_10 a o 10 5000

for \<dataset\>=laion_10, *turn on* CAS, \<top-k>\=10, \<ef_search\>=5000.

### Recommended \<ef_search\> Setting

**laion_4~laion_10**

300-20000 for \<CAS\>=o

## Dynamic Update Performance

Firstly, download file **dataset_and_indexfiles_for_update.tar** (the data and indexfiles containing in this tar file is necessary for all experiments in the paper) from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/IQBuREtfiHd4RK-e-DRzfimtARmX0qbkxks1vZ6pObn7-lc?e=t2Cjb9 and then decompress it:

    tar -xf dataset_and_indexfiles_for_update.tar
    mv dataset_and_indexfiles_for_update/data/* data/
    mv dataset_and_indexfiles_for_update/indexfiles/* indexfiles/

Notice: another 170GB and more storage space.

Secondly, compile test_ragtree_update_ada.cpp and test_ragtree_update_bruteforce.cpp

    cd test_ragtree_update
    ./compile_test_ragtree_update_ada.sh
    ./compile_test_ragtree_update_bruteforce.sh

Thirdly, running index update of **DeltaUpdate** and **ImmediateUpdate** by

    ./run_ragtree_update_ada.sh u <dataset> <update version> a
    ./run_ragtree_update_bruteforce.sh u <dataset> <update version> a

For example,

    ./run_ragtree_update_ada.sh u laion u_20 a

for **DeltaUpdate** on \<dataset\>=laion with \<update version\>=u_20 (20% data updates). And

    ./run_ragtree_update_bruteforce.sh u laion u_10 a

for **ImmediateUpdate** on \<dataset\>=laion with \<update version\>=u_10 (10% data updates).

Afterwards, running query processing on the updated index of **DeltaUpdate** by

    ./run_ragtree_update_ada.sh 1 <dataset> <update version> a <CAS> <top-k> <ef_search>

And running query processing on the updated index of **ImmediateUpdate** by

    ./run_ragtree_update_bruteforce.sh 1 <dataset> <update version> a <CAS> <top-k> <ef_search>

## Index Construction

Firstly,

    cd test_ragtree

Afterwards, run RaG-Tree index construction by

    ./run_ragtree_s.sh f <dataset> a

For example,

    ./run_ragtree_s.sh f dblp a

for \<dataset\>=dblp. And

    ./run_ragtree_s.sh f msmarco a

for \<dataset\>=msmarco. And

    ./run_ragtree_s.sh f laion a

for \<dataset\>=laion.

