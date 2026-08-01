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
#include <cstring>

// #include "data_query_vectorization.h"
// #include "hnswlib/hnswlib.h"
// #include "data_query_ori_v1.h"
#include "data_query_u.h"
// #include "rtree_nsw_64_v128_v256_v512_optv3.hpp"
#include "ragtree_update_bruteforce.hpp"

inline bool is_dataupdate_mode(const char *mode)
{
    return strcmp(mode, "u") == 0;
}

inline bool is_store_check_mode(const char *mode)
{
    return strcmp(mode, "c") == 0;
}

inline bool is_query_mode(const char *mode)
{
    return strcmp(mode, "1") == 0;
}

inline void print_usage(const char *program)
{
    cerr << "Mistake paras. The command should be:" << endl;
    cerr << program << " c dataset updatename" << endl;
    cerr << program << " u dataset updatename a/d [updates_path updated_index_path max_updates]" << endl;
    cerr << program << " 1 dataset updatename a/d o/d topk ef_search [updates_path updated_index_path max_updates]" << endl;
}

inline size_t parse_size_t_arg(const char *arg)
{
    stringstream ss;
    ss << arg;
    size_t value = 0;
    ss >> value;
    return value;
}

inline size_t parse_update_limit_arg(const char *arg)
{
    if (strcmp(arg, "all") == 0 || strcmp(arg, "ALL") == 0 || strcmp(arg, "-1") == 0)
        return numeric_limits<size_t>::max();
    return parse_size_t_arg(arg);
}

inline void ensure_parent_dir_exists(const char *path)
{
    filesystem::path p(path);
    filesystem::path parent = p.parent_path();
    if (!parent.empty())
        filesystem::create_directories(parent);
}

// 检查 Data / UpdateSet / workload 的读取和前若干条内容。
void store_check(const char *data_path,
                 const char *updates_path,
                 const char *workload_path,
                 size_t head_n)
{
    cerr << "Loading dataset " << data_path << " ..." << endl;
    Dataset data;
    data.load(data_path);
    pair<size_t, pair<size_t, size_t> > data_size = data.get_size();
    size_t n = data_size.first, d = data_size.second.first, m = data_size.second.second;
    cerr << "Data n=" << n << ", d=" << d << ", m=" << m << endl;

    cerr << "Loading updateset " << updates_path << " ..." << endl;
    UpdateSet updates;
    updates.load(updates_path);
    pair<size_t, pair<size_t, size_t> > updates_size = updates.get_size();
    size_t updates_n = updates_size.first;
    size_t updates_d = updates_size.second.first;
    size_t updates_m = updates_size.second.second;
    cerr << "UpdateSet n=" << updates_n << ", d=" << updates_d << ", m=" << updates_m << endl;

    cerr << "Loading query workload " << workload_path << " ..." << endl;
    vector<Query> *wkld = load_workload(workload_path);
    size_t wkld_n = wkld->size();
    cerr << "Workload n=" << wkld_n << endl;

    size_t data_head_n = min(head_n, n);
    size_t updates_head_n = min(head_n, updates_n);
    size_t workload_head_n = min(head_n, wkld_n);

    cout << endl << "Data vectors head=" << data_head_n << endl;
    for (size_t i = 0; i < data_head_n; i++)
    {
        float *veci = data.get_vector_i(i);
        cout << "data" << i << " vector=[";
        for (size_t j = 0; j < d; j++)
            cout << *(veci + j) << ", ";
        cout << "]" << endl;
    }

    cout << endl << "Data attributes head=" << data_head_n << endl;
    for (size_t i = 0; i < data_head_n; i++)
    {
        float *datai = data.get_attributes_i(i);
        cout << "data" << i << " attributes=(";
        for (size_t j = 0; j < m; j++)
            cout << fixed << *(datai + j) << ", ";
        cout << ')' << endl;
    }

    cout << endl << "UpdateSet head=" << updates_head_n << endl;
    for (size_t i = 0; i < updates_head_n; i++)
    {
        int update_type = updates.get_update_type(i);
        cout << "update" << i << " op=" << updates.get_op_i(i)
             << " type=" << update_type;
        if (update_type == 1)
        {
            float *up_vec = updates.get_vector_i(i);
            float *up_attr = updates.get_attributes_i(i);
            cout << " add vector=[";
            for (size_t j = 0; j < updates_d; j++)
                cout << *(up_vec + j) << ", ";
            cout << "] attributes=(";
            for (size_t j = 0; j < updates_m; j++)
                cout << fixed << *(up_attr + j) << ", ";
            cout << ')';
        }
        else if (update_type == -1)
            cout << " delete_id=" << updates.get_delete_id(i);
        else
            cout << " abnormal";
        cout << endl;
    }

    cout << endl << "Workload head=" << workload_head_n << endl;
    for (size_t i = 0; i < workload_head_n; i++)
    {
        float *qi_vec = (*wkld)[i].get_vector();
        cout << "query" << i << " vector=[";
        for (size_t j = 0; j < d; j++)
            cout << *(qi_vec + j) << ", ";
        cout << "] ";
        const float *qi_predlow = (*wkld)[i].get_predlow_data();
        const float *qi_predhigh = (*wkld)[i].get_predhigh_data();
        for (size_t j = 0; j < m; j++)
            cout << "Col" << j << " in range [" << fixed << *(qi_predlow + j)
                 << "," << fixed << *(qi_predhigh + j) << "]; ";
        cout << endl;
    }
}

// 数据更新：读取原始 Data 和 UpdateSet，加载原始索引，执行 update，只保存 updated index。
void dataupdate_rtreenewindex(const char *data_path,
                              const char *updates_path,
                              const char *model_path,
                              const char *updated_model_path,
                              size_t max_updates = numeric_limits<size_t>::max())
{
    cerr << "Loading dataset " << data_path << " ..." << endl;
    Dataset dataset;
    dataset.load(data_path);
    auto [dataset_n, dataset_dm] = dataset.get_size();
    auto [dataset_d, dataset_m] = dataset_dm;
    cerr << "Dataset n=" << dataset_n << ", d=" << dataset_d << ", m=" << dataset_m << endl;

    cerr << "Loading updateset " << updates_path << " ..." << endl;
    UpdateSet updates;
    updates.load(updates_path);
    auto [updates_n, updates_dm] = updates.get_size();
    auto [updates_d, updates_m] = updates_dm;
    if (updates_d != dataset_d || updates_m != dataset_m)
    {
        cerr << "Mistake updateset. vector/attributes dim mismatch: data=("
             << dataset_d << ',' << dataset_m << "), updates=("
             << updates_d << ',' << updates_m << ')' << endl;
        exit(-1);
    }

    double model_size = folder_size(model_path) / 1024.0 / 1024.0 / 1024.0;
    cerr << "Loading original index " << model_path << " (" << model_size << "GB) ..." << endl;
    RTreeNSW<Dataset> index(dataset, 2, 1);
    index.load_tree_binary(model_path, false);

    const size_t updates_to_apply = min(max_updates, updates_n);
    cerr << "Applying updates " << updates_to_apply << "/" << updates_n << " ..." << endl;
    long long start = clock();
    size_t add_count = 0, delete_count = 0;
    for (size_t i = 0; i < updates_to_apply; i++)
    {
        int update_type = updates.get_update_type(i);
        if (update_type == 1)
        {
            size_t id = dataset.add_record(updates.get_vector_i(i), updates.get_attributes_i(i));
            index.add_record(id);
            add_count++;
        }
        else if (update_type == -1)
        {
            index.delete_record(updates.get_delete_id(i));
            delete_count++;
        }
        else
        {
            cerr << "Mistake updateset. abnormal update type at statement " << i
                 << ", op=" << updates.get_op_i(i) << endl;
            exit(-1);
        }
    }
    long long end = clock();
    cerr << "Update Time: " << (double)(end - start) / CLOCKS_PER_SEC << " sec." << endl;
    cerr << "Original n=" << dataset_n << ", updated n=" << dataset.get_size().first
         << ", add=" << add_count
         << ", delete=" << delete_count << endl;

    cerr << "Dumping updated index to " << updated_model_path << " ..." << endl;
    index.saveupdatedindex(updated_model_path);
    cerr << "Done." << endl;
    cerr << "Update Time: " << (double)(end - start) / CLOCKS_PER_SEC << " sec." << endl;
    cerr << "Original n=" << dataset_n << ", updated n=" << dataset.get_size().first
         << ", add=" << add_count
         << ", delete=" << delete_count << endl;
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

void online_rtreenewindex_algo1(const char *data_path,
                                const char *updates_path,
                                const char *workload_path,
                                const char *model_path,
                                const size_t k,
                                size_t ef_search,
                                char ada,
                                char opt,
                                bool debug_flag = false,
                                size_t max_updates = numeric_limits<size_t>::max())
{
    cerr << "Loading dataset " << data_path << " ..." << endl;
    Dataset dataset;
    dataset.load(data_path);
    auto [dataset_n, dataset_dm] = dataset.get_size();
    auto [dataset_d, dataset_m] = dataset_dm;
    cerr << "Dataset n=" << dataset_n << ", d=" << dataset_d << ", m=" << dataset_m << endl;

    cerr << "Loading updateset " << updates_path << " ..." << endl;
    UpdateSet updates;
    updates.load(updates_path);
    auto [updates_n, updates_dm] = updates.get_size();
    auto [updates_d, updates_m] = updates_dm;
    if (updates_d != dataset_d || updates_m != dataset_m)
    {
        cerr << "Mistake updateset. vector/attributes dim mismatch: data=("
             << dataset_d << ',' << dataset_m << "), updates=("
             << updates_d << ',' << updates_m << ')' << endl;
        exit(-1);
    }

    const size_t updates_to_apply = min(max_updates, updates_n);
    size_t add_count = 0, delete_count = 0;
    for (size_t i = 0; i < updates_to_apply; i++)
    {
        int update_type = updates.get_update_type(i);
        if (update_type == 1)
        {
            dataset.add_record(updates.get_vector_i(i), updates.get_attributes_i(i));
            add_count++;
        }
        else if (update_type == -1)
        {
            delete_count++;
        }
        else
        {
            cerr << "Mistake updateset. abnormal update type at statement " << i
                 << ", op=" << updates.get_op_i(i) << endl;
            exit(-1);
        }
    }
    cerr << "Original n=" << dataset_n << ", updated n=" << dataset.get_size().first
         << ", add=" << add_count
         << ", delete=" << delete_count << endl;
    auto [upddataset_n, upddataset_dm] = dataset.get_size();
    auto [upddataset_d, upddataset_m] = upddataset_dm;
    cerr << "UpatedDataset n=" << upddataset_n << ", d=" << upddataset_d << ", m=" << upddataset_m << endl;

    cerr << "Loading query workload " << workload_path << " ..." << endl;
    vector<Query> *wkld = load_workload(workload_path);
    size_t wkld_n = wkld->size();

    //double model_size = filesystem::file_size(model_path) / 1024.0 / 1024.0 / 1024.0;
    double model_size = folder_size(model_path) / 1024.0 / 1024.0 / 1024.0;
    cerr << "Loading updated index " << model_path << " (" << model_size << "GB) ..." << endl;
    RTreeNSW<Dataset> loaded_tree(dataset, 2, 1);
    loaded_tree.set_hnsw_ef_search(ef_search);
    // exit(-1);
    // if(ads)
    //     loaded_tree.set_ads_tau(0.99);
    // else
    //     loaded_tree.set_ads_tau(0);
    loaded_tree.loadupdatedindex(model_path, false);
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
    const int mode_arg = 1;
    const int dataset_arg = 2;
    const int update_arg = 3;
    const int flag_arg = 4;

    if (argc <= update_arg)
    {
        print_usage(argv[0]);
        exit(-1);
    }
    char *mode = argv[mode_arg];
    if (!is_store_check_mode(mode) && !is_query_mode(mode) && !is_dataupdate_mode(mode))
    {
        cerr << "Mistake paras. mode should be c/u/1" << endl;
        print_usage(argv[0]);
        exit(-1);
    }
    string dataset_name = argv[dataset_arg];
    string update_name = argv[update_arg];
    string dataset_file_prefix = dataset_name;
    string updated_file_prefix = dataset_name + "_" + update_name;
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
    ostringstream data_path, updates_path, workload_path;
    data_path << "../data/" << dataset_dir << "/" << dataset_file_prefix << "_data.npz";
    updates_path << "../data/" << updated_file_prefix << "/" << updated_file_prefix << "_updateset.npz";
    workload_path << "../data/" << dataset_dir << "/" << dataset_file_prefix << "_selrandom_query.npz";
    
#ifdef HNSW_U_BRUTEFORCE
    cerr << "ragtree_u_bruteforce";
#else
    cerr << "NOT ragtree_u_bruteforce, ERROR!" << endl;
    return -1;
#endif
// #ifdef RTREE_NSW_USE_HNSWLIB
//     cerr << "_RTREE_NSW_USE_HNSWLIB";
// #endif
    cerr << endl;
    if (is_store_check_mode(mode))
        store_check(data_path.str().c_str(), updates_path.str().c_str(), workload_path.str().c_str(), 5);
    else
    {
        if (argc <= flag_arg)
        {
            print_usage(argv[0]);
            exit(-1);
        }
        ostringstream index_save_path, updated_index_save_path;
        char flag_ada = argv[flag_arg][0];
        if (flag_ada != 'a' && flag_ada != 'd')
        {
            cerr << "Mistake paras. a/d flag should be a or d" << endl;
            exit(-1);
        }
        if (flag_ada == 'a')
        {
            index_save_path << "../indexfiles/rtreehnsw_s_a_" << dataset_file_prefix;
            updated_index_save_path << "../indexfiles/rtreehnsw_u_all_a_" << updated_file_prefix;
        }
        else
        {
            index_save_path << "../indexfiles/rtreehnsw_s_d_" << dataset_file_prefix;
            updated_index_save_path << "../indexfiles/rtreehnsw_u_all_d_" << updated_file_prefix;
        }
        if (is_dataupdate_mode(mode))
        {
            string actual_updates_path = updates_path.str();
            string actual_updated_index_path = updated_index_save_path.str();
            size_t max_updates = numeric_limits<size_t>::max();

            if (argc > flag_arg + 1)
                actual_updates_path = argv[flag_arg + 1];
            if (argc > flag_arg + 2)
                actual_updated_index_path = argv[flag_arg + 2];
            if (argc > flag_arg + 3)
                max_updates = parse_update_limit_arg(argv[flag_arg + 3]);

            dataupdate_rtreenewindex(data_path.str().c_str(),
                                     actual_updates_path.c_str(),
                                     index_save_path.str().c_str(),
                                     actual_updated_index_path.c_str(),
                                     max_updates);
        }
        else if (is_query_mode(mode))
        {
            if (argc <= flag_arg + 3)
            {
                print_usage(argv[0]);
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
            string query_updates_path = updates_path.str();
            string query_index_path = updated_index_save_path.str();
            if (argc > flag_arg + 4)
                query_updates_path = argv[flag_arg + 4];
            if (argc > flag_arg + 5)
                query_index_path = argv[flag_arg + 5];
            size_t max_updates = numeric_limits<size_t>::max();
            if (argc > flag_arg + 6)
                max_updates = parse_update_limit_arg(argv[flag_arg + 6]);
            online_rtreenewindex_algo1(data_path.str().c_str(),
                                       query_updates_path.c_str(),
                                       workload_path.str().c_str(),
                                       query_index_path.c_str(),
                                       k,
                                       ef_search,
                                       flag_ada,
                                       flag_opt,
                                       false,
                                       max_updates);
            // }
            // else
            //     online_rtreenewindex_algo1(data_path.str().c_str(), workload_path.str().c_str(), index_save_path.str().c_str(), k);
        }
    }

    return 0;
}
