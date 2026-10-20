// basal-logits: the code logits of each port-check item with llama.cpp, in DeviceCheck's row format, so that gate.py
// compares a GGUF on a phone's GPU the same way as a LiteRT variant.
//
//   basal-logits -m basal-2.0-mini-Q8_0.gguf -c code_ids.txt -i items_ids.jsonl -o out.jsonl [-n 300] [-ngl 99]
//                [-t 4] [-ub 512] [-ctx 8192] [-fa 0|1]
//
// Items carry the prompt's token ids (with <s>), so llama.cpp's tokenizer is not involved. The cache keeps the start
// a prompt shares with the one before, as the app keeps each language's prompt start: "ms" is the rest of the prefill.
#include "llama.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void usage() {
    fprintf(stderr, "usage: basal-logits -m model.gguf -c code_ids.txt -i items.jsonl -o out.jsonl "
                    "[-n N] [-ngl L] [-t threads] [-ub ubatch] [-ctx n_ctx]\n");
    exit(2);
}

int main(int argc, char ** argv) {
    std::string model_path, codes_path, items_path, out_path;
    int n_items = 0, ngl = 99, threads = 4, ubatch = 512, n_ctx = 8192, fa = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (i + 1 >= argc) usage();
        const char * v = argv[++i];
        if      (a == "-m")   model_path = v;
        else if (a == "-c")   codes_path = v;
        else if (a == "-i")   items_path = v;
        else if (a == "-o")   out_path = v;
        else if (a == "-n")   n_items = atoi(v);
        else if (a == "-ngl") ngl = atoi(v);
        else if (a == "-t")   threads = atoi(v);
        else if (a == "-ub")  ubatch = atoi(v);
        else if (a == "-ctx") n_ctx = atoi(v);
        else if (a == "-fa")  fa = atoi(v);
        else usage();
    }
    if (model_path.empty() || codes_path.empty() || items_path.empty() || out_path.empty()) usage();

    std::vector<int> codes;
    {
        std::ifstream f(codes_path);
        for (int id; f >> id;) codes.push_back(id);
    }

    llama_backend_init();
    ggml_backend_load_all();
    auto t0 = std::chrono::steady_clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = ubatch;
    cp.n_ubatch = ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    // off by default: with "auto" the cache keeps V untransposed, and where the backend then declines flash attention
    // (OpenCL on Adreno 7xx) every layer copies V to a contiguous tensor that misses the Adreno KQV kernel
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int id : codes) if (id < 0 || id >= n_vocab) { fprintf(stderr, "code id %d outside the vocabulary\n", id); return 1; }
    long load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "loaded in %ld ms\n", load_ms);

    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::ifstream in(items_path);
    std::ofstream out(out_path);
    std::vector<llama_token> prev;
    std::string line;
    int done = 0;
    long total_ms = 0;
    while (std::getline(in, line) && (n_items <= 0 || done < n_items)) {
        if (line.empty()) continue;
        json it = json::parse(line);
        std::vector<llama_token> ids = it["ids"].get<std::vector<llama_token>>();
        const int n = (int) ids.size();
        if (n > n_ctx) { done++; continue; }
        // the cached start shared with the previous prompt; the last token always runs, for its logits
        int keep = 0;
        while (keep < (int) prev.size() && keep < n - 1 && prev[keep] == ids[keep]) keep++;
        auto t = std::chrono::steady_clock::now();
        llama_memory_seq_rm(mem, 0, keep, -1);
        for (int p = keep; p < n; p += ubatch) {
            batch.n_tokens = 0;
            for (int j = p; j < n && j < p + ubatch; j++) {
                const int b = batch.n_tokens++;
                batch.token[b] = ids[j];
                batch.pos[b] = j;
                batch.n_seq_id[b] = 1;
                batch.seq_id[b][0] = 0;
                batch.logits[b] = j == n - 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at item %d\n", done); return 1; }
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
        total_ms += ms;
        prev = ids;

        const size_t k = it["keys"].size();
        std::vector<float> row(k);
        for (size_t j = 0; j < k; j++) row[j] = logits[codes[j]];
        it.erase("ids");
        it["tokens"] = n;
        it["ms"] = ms;
        it["cached"] = keep;
        it["logits"] = row;
        out << it.dump() << "\n";
        out.flush();
        done++;
        if (done % 50 == 0) fprintf(stderr, "%d items, %ld ms/item\n", done, total_ms / done);
    }
    fprintf(stderr, "done: %d items, %ld ms/item, load %ld ms\n", done, total_ms / (done ? done : 1), load_ms);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
