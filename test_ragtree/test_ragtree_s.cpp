// #include <iostream>
// #include <NumCpp.hpp>
// #include <cnpy.h>

// using namespace std;

// template <typename T>
// inline nc::NdArray<T> load_datanpy(const char *path)
// {
//         cnpy::NpyArray datanpy = cnpy::npy_load(path);
//         size_t n=datanpy.shape[0], c=datanpy.shape[1];
//         nc::NdArray<T> data(n, c);
//         T* p = datanpy.data<T>();
//         copy(p, p+n*c, data.data());
//         /*
//         for(size_t i = 0; i < n; i++)
//                 for(size_t j = 0; j < c; j++)
//                         data(i, j) = p[i * c + j];*/
//         return data;
// }

#include <iomanip>
#include <ctime>
#include <filesystem>
#include <sstream>

// #include "data_query_vectorization.h"
// #include "hnswlib/hnswlib.h"
#include "data_query_ori_v1.h"
// #include "rtree_nsw_64_v128_v256_v512_optv3.hpp"
#include "ragtree_s.hpp"

//检查data_query.h的正确性
void store_check(const char *data_path, const char *workload_path, size_t head_n)
{
    Dataset *data = new Dataset();
    data->load(data_path);
    pair<size_t, pair<size_t, size_t> > data_size = data->get_size();
    size_t n=data_size.first, d=data_size.second.first, m=data_size.second.second;
    cerr << n << ' ' << d << ' ' << m << endl;

    vector<Query> *wkld = load_workload(workload_path);
    size_t wkld_n = wkld->size();
    cerr << wkld_n << endl;

    // size_t head_n = 9;
    
    for(size_t i = 0; i < head_n; i++)
    {
        float *veci = data->get_vector_i(i);
        cout << '[';
        for(size_t j = 0; j < d; j++)
            cout << *(veci+j) << ", ";
        cout << "]" << endl;
    }
    cout << endl;

    // for(size_t i = 0; i < head_n; i++)
    // {
    //     cout << '(';
    //     for(size_t j = 0; j < m; j++)
    //         cout << fixed << data->get_attributes_j_i(j, i) << ", ";
    //     cout << ')' << endl;
    // }
    // cout << endl;

    // for(size_t i = 0; i < head_n; i++)
    // {
    //     float *qi_vec = (*wkld)[i].get_vector();
    //     cout << '[';
    //     for(size_t j = 0; j < d; j++)
    //         cout << *(qi_vec+j) << ", ";
    //     cout << "] ";
    //     for(size_t j = 0; j < m; j++)
    //         cout << "Col" << j << " in range [" << fixed << (*wkld)[i].get_predlow_j(j) << "," << fixed << (*wkld)[i].get_predhigh_j(j) << "]; ";
    //     cout << endl;
    // }

    cout << endl << "Vectorization" << endl << endl;

    for(size_t i = 0; i < head_n; i++)
    {
        float *datai = data->get_attributes_i(i);
        cout << '(';
        for(size_t j = 0; j < m; j++)
            cout << fixed << *(datai+j) << ", ";
        cout << ')' << endl;
    }
    cout << endl;

    for(size_t i = 0; i < head_n; i++)
    {
        float *qi_vec = (*wkld)[i].get_vector();
        cout << '[';
        for(size_t j = 0; j < d; j++)
            cout << *(qi_vec+j) << ", ";
        cout << "] ";
        const float *qi_predlow = (*wkld)[i].get_predlow_data(), *qi_predhigh = (*wkld)[i].get_predhigh_data();
        for(size_t j = 0; j < m; j++)
            cout << "Col" << j << " in range [" << fixed << *(qi_predlow+j) << "," << fixed << *(qi_predhigh+j) << "]; ";
        cout << endl;
    }
}

//离线构建纯RTree索引，给定数据路径和模型保存路径
void offline_rtreensw_construction(const char *data_path, const char *model_save_path, char flag_ada = 'd')
{
    cerr << "Loading dataset " << data_path << " ..." << endl;
    Dataset dataset;
    dataset.load(data_path);
    auto [dataset_n, dataset_dm] = dataset.get_size();
    auto [dataset_d, dataset_m] = dataset_dm;

    long long start = clock();
    size_t max_c = ceil(1000.0 * pow(dataset_n/1000000.0, (9.0/4.0)) * pow(2, 0.5*(dataset_m-1.0)));
    size_t ce_max_c = min(max_c * 1.0, ceil(0.001 * dataset_n));
    RTreeNSW<Dataset> tree(dataset, ce_max_c, ce_max_c >> 2, ce_max_c, ce_max_c >> 2);
    tree.set_hnsw_ef_construction(128);
    tree.set_hnsw_max_neighbors(16);
    // cerr << "SET ef_construction = " << tree.hnsw_ef_construction_ << endl;
    // cerr << "SET max_neighbors = " << tree.hnsw_max_neighbors_ << endl;
    cerr << "Building RTreeNSW (" << tree.options_.max_capacity << ", " << tree.options_.min_capacity << ", " << tree.options_.ce_leaf_max_capacity << ", " << tree.options_.ce_leaf_min_capacity << ") ..." << endl;
    if (flag_ada == 'a')
        tree.build_ada(0.1);
    else
        tree.build();
    long long end = clock();
    cerr << "Construction Time: " << (double)(end - start) / CLOCKS_PER_SEC << " sec." << endl;
    // string modelpathprefix = "rtreensw_", modeldata = data_path, modelpathsuffix = ".bin";
    // string modelpath = modelpathprefix + modeldata + modelpathsuffix;
    cerr << "Dumping to " << model_save_path << " ..." << endl;
    tree.save_tree_binary(model_save_path);
    cerr << "Done." << endl;
    cerr << "Construction Time: " << (double)(end - start) / CLOCKS_PER_SEC << " sec." << endl;

    // RTreeNSW<MockDataset> loaded_tree(dataset, 2, 1);
    // loaded_tree.load_tree_binary("/private/tmp/rtree_range_query_check.bin");

    // MockQuery query;
    // vector<size_t> a = tree.range_query_stop_at_branch(query);
    // vector<size_t> b = tree.range_query_leaf_scan(query);
    // vector<size_t> loaded_a = loaded_tree.range_query_stop_at_branch(query);
    // vector<size_t> loaded_b = loaded_tree.range_query_leaf_scan(query);
    // return a.empty() || b.empty() || a != loaded_a || b != loaded_b;
}

//纯RTree索引在线查询算法2，给定数据路径、查询负载路径和模型路径
// void online_rtreeindex_range_queries_algo2(const char *data_path, const char *workload_path, const char *model_path)
// {
//     cerr << "Loading dataset " << data_path << " ..." << endl;
//     Dataset dataset;
//     dataset.load(data_path);

//     cerr << "Loading query workload " << workload_path << " ..." << endl;
//     vector<Query> *wkld = load_workload(workload_path);
//     size_t wkld_n = wkld->size();

//     double model_size = filesystem::file_size(model_path) / 1024.0 / 1024.0 / 1024.0;
//     cerr << "Loading " << model_path << " (" << model_size << "GB) ..." << endl;
//     RTreeNSW<Dataset> loaded_tree(dataset, 2, 1);
//     loaded_tree.load_tree_binary(model_path);
//     cerr << "Loaded with options = " << loaded_tree.options_.max_capacity << ", " << loaded_tree.options_.min_capacity << endl;
    
//     cerr << "Querying ..." << endl;
//     vector<vector<size_t> > all_res(wkld_n);
//     double total_time = 0.0;
//     for(size_t qi = 0; qi < wkld_n; qi++)
//     {
//         cerr << "query" << qi << endl;
//         long long start = clock();
//         vector<size_t> res = loaded_tree.range_query_leaf_scan(wkld->operator[](qi));
//         long long end = clock();
//         total_time += (end - start) * 1000.0L / CLOCKS_PER_SEC;
//         all_res[qi] = move(res);
//     }
//     double avg_time = total_time / wkld_n;
//     cerr << wkld_n << "queries " << avg_time << "ms/query" << " QPS=" << 1000.0L / avg_time << endl;
    
//     cerr << "Dumping result ..." << endl;
//     cout << "result\n";
//     for(size_t qi = 0; qi < wkld_n; qi++)
//     {
//         for(vector<size_t>::iterator t = all_res[qi].begin(); t != all_res[qi].end(); ++t)
//             cout << *t << ' ';
//         cout << "\n";
//     }
//     cerr << "Done.";
// }

// //RTreeNSW混合检索，RTree索引Range Filtering在线查询算法1，给定数据路径、查询负载路径和模型路径
// void online_rtreenewindex_algo1(const char *data_path, const char *workload_path, const char *model_path, const size_t k)
// {
//     cerr << "Loading dataset " << data_path << " ..." << endl;
//     Dataset dataset;
//     dataset.load(data_path);

//     cerr << "Loading query workload " << workload_path << " ..." << endl;
//     vector<Query> *wkld = load_workload(workload_path);
//     size_t wkld_n = wkld->size();

//     //double model_size = filesystem::file_size(model_path) / 1024.0 / 1024.0 / 1024.0;
//     double model_size = folder_size(model_path) / 1024.0 / 1024.0 / 1024.0;
//     cerr << "Loading " << model_path << " (" << model_size << "GB) ..." << endl;
//     RTreeNSW<Dataset> loaded_tree(dataset, 2, 1);
//     loaded_tree.load_tree_binary(model_path);
//     cerr << "Loaded with options = " << loaded_tree.options_.max_capacity << ", " << loaded_tree.options_.min_capacity << "ef_search=" << loaded_tree.hnsw_ef_search_ << endl;
    
//     cerr << "Querying ..." << endl;
//     vector<vector<pair<size_t, float> > > all_res(wkld_n);
//     double total_time = 0.0;
//     for(size_t qi = 0; qi < wkld_n; qi++)
//     {
//         // cerr << "query" << qi << endl;
//         long long start = clock();
//         vector<pair<size_t, float> > res = loaded_tree.rfanns_query_topk(wkld->operator[](qi), k);
//         long long end = clock();
//         total_time += (end - start) * 1000.0L / CLOCKS_PER_SEC;
//         all_res[qi] = move(res);
//     }
//     double avg_time = total_time / wkld_n;
//     cerr << wkld_n << "queries " << avg_time << "ms/query" << " QPS=" << 1000.0L / avg_time << endl;
    
//     cerr << "Dumping result ..." << endl;
//     cout << "result\n";
//     for(size_t qi = 0; qi < wkld_n; qi++)
//     {
//         for(vector<pair<size_t, float> >::iterator t = all_res[qi].begin(); t != all_res[qi].end(); ++t)
//             cout << t->first << ' ';
//         cout << "\n";
//     }
//     cerr << "Done." << endl;
// }

void online_rtreenewindex_algo1(const char *data_path, const char *workload_path, const char *model_path, const size_t k, size_t ef_search, char ada, char opt, bool debug_flag = false)
{
    cerr << "Loading dataset " << data_path << " ..." << endl;
    Dataset dataset;
    dataset.load(data_path);

    cerr << "Loading query workload " << workload_path << " ..." << endl;
    vector<Query> *wkld = load_workload(workload_path);
    size_t wkld_n = wkld->size();

    //double model_size = filesystem::file_size(model_path) / 1024.0 / 1024.0 / 1024.0;
    double model_size = folder_size(model_path) / 1024.0 / 1024.0 / 1024.0;
    cerr << "Loading " << model_path << " (" << model_size << "GB) ..." << endl;
    RTreeNSW<Dataset> loaded_tree(dataset, 2, 1);
    loaded_tree.set_hnsw_ef_search(ef_search);
    // exit(-1);
    // if(ads)
    //     loaded_tree.set_ads_tau(0.99);
    // else
    //     loaded_tree.set_ads_tau(0);
    loaded_tree.load_tree_binary(model_path);
    cerr << "Loaded with options=" << loaded_tree.options_.max_capacity << ", " << loaded_tree.options_.min_capacity << ' ' << ada << " ef_search=" << loaded_tree.hnsw_ef_search_setting_ << endl;
    
    cerr << "Querying(" << opt << ") ..." << endl;
    bool flag_ada = (ada == 'a');
    vector<vector<pair<size_t, float> > > all_res(wkld_n);
    long long start, end;
    // double total_time = 0.0;
    if(opt == 'o')
    {
        start = clock();
        for(size_t qi = 0; qi < wkld_n; qi++)
        {
            //cerr << "query" << qi << endl;
            all_res[qi] = loaded_tree.rfanns_query_topk_opt(wkld->operator[](qi), k, debug_flag);
#ifdef DP_DEBUG
            cerr << "ActualPlan:" << endl;
            all_res[qi] = loaded_tree.rfanns_query_topk_opt(wkld->operator[](qi), k, debug_flag, true);
            cerr << endl;
#endif
            // cerr << "ActualCostPlan";
            // all_res[qi] = loaded_tree.rfanns_query_topk_opt(wkld->operator[](qi), k, debug_flag, true);
            // cerr << endl;
            // all_res[qi] = loaded_tree.rfanns_query_topk_opt(wkld->operator[](qi), k, debug_flag);
            // all_res[qi] = loaded_tree.rfanns_query_topk_opt(wkld->operator[](qi), k, debug_flag);
        }
        end = clock();
    }
    else
    {
        start = clock();
        for(size_t qi = 0; qi < wkld_n; qi++)
        {
            //cerr << "query" << qi << endl;
            // all_res[qi] = loaded_tree.rfanns_query_topk_greedy(wkld->operator[](qi), k, debug_flag);
            // all_res[qi] = loaded_tree.rfanns_query_topk_greedy(wkld->operator[](qi), k, debug_flag);
            all_res[qi] = loaded_tree.rfanns_query_topk_greedy(wkld->operator[](qi), k, debug_flag);
            // all_res[qi] = loaded_tree.rfanns_query_topk_greedy_ex(wkld->operator[](qi), k, debug_flag);
            // cerr << "ActualPlan:" << endl;
            // all_res[qi] = loaded_tree.rfanns_query_topk_greedy(wkld->operator[](qi), k, debug_flag, true);
            // cerr << endl;
        }
        end = clock();
    }
    double total_time = (end - start) * 1000.0L / CLOCKS_PER_SEC;
    double avg_time = total_time / (wkld_n * 1.0);
    cerr << wkld_n << "queries " << avg_time << "ms/query" << " QPS=" << 1000.0L / avg_time << endl;
    
    cerr << "Dumping result ..." << endl;
    cout << "result\n";
    for(size_t qi = 0; qi < wkld_n; qi++)
    {
        for(vector<pair<size_t, float> >::iterator t = all_res[qi].begin(); t != all_res[qi].end(); ++t)
            cout << t->first << ' ';
        cout << "\n";
    }
    cerr << "Done." << endl;
}

int main(int argc, char* argv[])
{
    // char *attr_n = argv[1];
    if (argc < 2)
    {
        cerr << "Mistake paras. The command should be ./program c/f/1 dataset [a/d] [o/d topk ef_search]" << endl;
        exit(-1);
    }
    char *mode = argv[1];
    if (mode[0] != 'c' && mode[0] != 'f' && mode[0] != '1')
    {
        cerr << "Mistake paras. mode should be c/f/1" << endl;
        exit(-1);
    }
    if (argc < 3)
    {
        cerr << "Mistake paras. dataset name is required" << endl;
        exit(-1);
    }
    string dataset_name = argv[2];
    const int flag_arg = 3;
    string dataset_file_prefix = dataset_name;
    string dataset_dir = dataset_name;
    //cnpy::NpyArray datanpy = cnpy::npy_load("laion.npy");
    // nc::NdArray<double> data = load_datanpy<double>("laion.npy");

    // cout << data.numRows() << endl;
    // cout << data.numCols() << endl;
    // auto row0 = data(0, data.cSlice());
    // cout << row0 << endl;

    // pair< nc::NdArray<double>*, nc::NdArray<double>* > data = load_datanpz<double>("laion.npz");
    // nc::NdArray<double> *data_vector = data.first;
    // nc::NdArray<double> *data_attributes = data.second;

    // cout << data_vector->numRows() << ',' << data_vector->numCols() << endl;
    // cout << data_attributes->numRows() << ',' << data_attributes->numCols() << endl;
    // auto vec_row0 = data_vector->operator()(0, data_vector->cSlice());
    // auto attributes_row0 = data_attributes->operator()(0, data_attributes->cSlice());
    // auto vec_rowlast1 = data_vector->operator()(data_vector->numRows()-1, data_vector->cSlice());
    // auto attributes_rowlast1 = data_attributes->operator()(data_attributes->numRows()-1, data_attributes->cSlice());

    // cout << vec_row0 << " , " << attributes_row0 << endl;
    // cout << vec_rowlast1 << " , " << attributes_rowlast1 << endl;

    // Dataset *data = new Dataset();
    // data->load("laion.npz");
    // pair<size_t, pair<size_t, size_t> > data_size = data->get_size();
    // size_t n=data_size.first, d=data_size.second.first, m=data_size.second.second;

    // // vector<float> *vec0 = data->get_vector_i(0);
    // // cout << '[';
    // // for(size_t j = 0; j < d; j++)
    // //     cout << vec0->operator[](j) << ", ";
    // // cout << "], (";
    // // for(size_t j = 0; j < m; j++)
    // //     cout << data->get_attributes_j_i(j, 0) << ", ";
    // // cout << ')' << endl;

    // // vector<float> *veclast1 = data->get_vector_i(n-1);
    // // cout << '[';
    // // for(size_t j = 0; j < d; j++)
    // //     cout << veclast1->operator[](j) << ", ";
    // // cout << "], (";
    // // for(size_t j = 0; j < m; j++)
    // //     cout << data->get_attributes_j_i(j, n-1) << ", ";
    // // cout << ')' << endl;

    // float *vec0 = data->get_vector_i(0);
    // cout << '[';
    // for(size_t j = 0; j < d; j++)
    //     cout << *(vec0+j) << ", ";
    // cout << "], (";
    // for(size_t j = 0; j < m; j++)
    //     cout << data->get_attributes_j_i(j, 0) << ", ";
    // cout << ')' << endl;

    // float *veclast1 = data->get_vector_i(n-1);
    // cout << '[';
    // for(size_t j = 0; j < d; j++)
    //     cout << *(veclast1+j) << ", ";
    // cout << "], (";
    // for(size_t j = 0; j < m; j++)
    //     cout << data->get_attributes_j_i(j, n-1) << ", ";
    // cout << ')' << endl;

    // data->check_attr_store_head(1);
    // data->check_attr_store_tail(1);

    // float *vec0 = data->get_vector_i(0);
    // cout << '[';
    // for(size_t j = 0; j < d; j++)
    //     cout << *(vec0+j) << ", ";
    // cout << "], (";
    // for(size_t j = 0; j < m; j++)
    //     cout << data->get_attributes_j_i(j, 0) << ", ";
    // cout << ')' << endl;

    // float *veclast1 = data->get_vector_i(n-1);
    // cout << '[';
    // for(size_t j = 0; j < d; j++)
    //     cout << *(veclast1+j) << ", ";
    // cout << "], (";
    // for(size_t j = 0; j < m; j++)
    //     cout << data->get_attributes_j_i(j, n-1) << ", ";
    // cout << ')' << endl;

    // data->check_attr_store_head(1);
    // data->check_attr_store_tail(1);

    //store_check();
    ostringstream data_path, workload_path;
    data_path << "../data/" << dataset_dir << "/" << dataset_file_prefix << "_data.npz";
    workload_path << "../data/" << dataset_dir << "/" << dataset_file_prefix << "_selrandom_query.npz";
    
#ifdef HNSW_S
    cerr << "ragtree_s";
#else
    cerr << "NOT ragtree_s, ERROR!" << endl;
    return -1;
#endif
// #ifdef RTREE_NSW_USE_HNSWLIB
//     cerr << "_RTREE_NSW_USE_HNSWLIB";
// #endif
    cerr << endl;
    if (mode[0] == 'c')
        store_check(data_path.str().c_str(), workload_path.str().c_str(), 3);
    else
    {
        if (argc <= flag_arg)
        {
            cerr << "Mistake paras. The command should be ./program f/1 dataset a/d" << endl;
            exit(-1);
        }
        ostringstream index_save_path;
        char flag_ada = argv[flag_arg][0];
        if (flag_ada != 'a' && flag_ada != 'd')
        {
            cerr << "Mistake paras. a/d flag should be a or d" << endl;
            exit(-1);
        }
        if (flag_ada == 'a')
            index_save_path << "../indexfiles/rtreehnsw_s_a_" << dataset_file_prefix;
        else
            index_save_path << "../indexfiles/rtreehnsw_s_d_" << dataset_file_prefix;
        if (mode[0] == 'f')
        {
            offline_rtreensw_construction(data_path.str().c_str(), index_save_path.str().c_str(), flag_ada);
        }
        else if (mode[0] == '1')
        {
            if (argc <= flag_arg + 3)
            {
                cerr << "Mistake paras. The command should be ./program 1 dataset a/d o/d topk ef_search" << endl;
                exit(-1);
            }
            char flag_opt = argv[flag_arg + 1][0];
            if (flag_opt != 'o' && flag_opt != 'd')
            {
                cerr << "Mistake paras. o/d flag should be o or d" << endl;
                exit(-1);
            }
            char *topk = argv[flag_arg + 2];
            stringstream sstopk;
            sstopk << topk;
            size_t k;
            sstopk >> k;
            cerr << "algo1 for KNN-top" << k << endl;
            // if (argc >= 5)
            // {
            char *s_ef_search = argv[flag_arg + 3];
            stringstream ssefsearch;
            ssefsearch << s_ef_search;
            size_t ef_search;
            ssefsearch >> ef_search;
            online_rtreenewindex_algo1(data_path.str().c_str(), workload_path.str().c_str(), index_save_path.str().c_str(), k, ef_search, flag_ada, flag_opt, false);
            // }
            // else
            //     online_rtreenewindex_algo1(data_path.str().c_str(), workload_path.str().c_str(), index_save_path.str().c_str(), k);
        }
    }

    return 0;
}
