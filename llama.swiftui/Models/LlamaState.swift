import Foundation
import os

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    @Published var serverRunning = false
    @Published var serverAddress = ""
    let httpServer = HTTPServer(port: 8080)

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(
            name: "Gemma-3-4B-IT (Q4_K_M, 2.9 GiB)",
            url: "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf?download=true",
            filename: "gemma-3-4b-it-Q4_K_M.gguf", status: "download"
        ),
        Model(
            name: "Qwen-3.5-4B (Q4_K_M, 2.7 GiB)",
            url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf?download=true",
            filename: "qwen3.5-4b-Q4_K_M.gguf", status: "download"
        ),
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    /// The model file's size, and roughly what it needs beside it while it answers (context cache, scratch buffers).
    private static func modelSize(_ modelUrl: URL) -> UInt64 {
        (try? FileManager.default.attributesOfItem(atPath: modelUrl.path())[.size] as? NSNumber)?.uint64Value ?? 0
    }
    private static let working: UInt64 = 600 * 1024 * 1024
    private static let headroom: UInt64 = 1024 * 1024 * 1024

    private static func gb(_ bytes: UInt64) -> String { String(format: "%.1f GB", Double(bytes) / 1_073_741_824) }

    func loadModel(modelUrl: URL?) throws {
        if let modelUrl {
            // The old model goes first: holding two at once is how a switch runs out of memory.
            llamaContext = nil
            // iPadOS closes an app that goes over its memory limit, and on the GPU the whole model counts against it once it
            // answers. A model that fits with a gigabyte to spare runs there; a bigger one runs on the CPU from its file,
            // which doesn't count, so only the context cache and buffers have to fit.
            let size = LlamaState.modelSize(modelUrl)
            let allowed = UInt64(os_proc_available_memory())
            let onGPU = allowed == 0 || size + LlamaState.working + LlamaState.headroom <= allowed
            if !onGPU && LlamaState.working > allowed {
                messageLog += "Not loaded: iPadOS lets this app use only \(LlamaState.gb(allowed)) right now. Close other apps and try again.\n"
                return
            }
            messageLog += onGPU
                ? "Loading model on the GPU (needs about \(LlamaState.gb(size + LlamaState.working)) of the \(LlamaState.gb(allowed)) this app may use)...\n"
                : "Too big for the GPU here (about \(LlamaState.gb(size + LlamaState.working)) of \(LlamaState.gb(allowed))): running it on the CPU, read from its file as needed. Slower, but within iPadOS's limit...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path(), onGPU: onGPU)
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(await llamaContext.n_len) / t_generation

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }

    // MARK: - HTTP Server

    func toggleServer() {
        if serverRunning {
            httpServer.stop()
            serverRunning = false
            serverAddress = ""
            messageLog += "Server stopped\n"
        } else {
            do {
                try httpServer.start(llamaState: self)
                serverRunning = true
                let ip = httpServer.getLocalIP()
                serverAddress = "http://\(ip):8080"
                messageLog += "Server started at \(serverAddress)\n"
            } catch {
                messageLog += "Failed to start server: \(error)\n"
            }
        }
    }

    /// One API answer: its text, how long the prompt and answer were, and whether it stopped at the length limit.
    struct APICompletion {
        var text: String = ""
        var promptTokens: Int = 0
        var completionTokens: Int = 0
        var finishReason: String = "stop"
        var error: String? = nil
    }

    // Requests take turns: two completions interleaving on one context would corrupt both.
    private let apiGate = APIGate()

    // Non-streaming completion for API use
    func completeForAPI(text: String, maxTokens: Int = 500, temperature: Float = 0.4) async -> APICompletion {
        guard let llamaContext else {
            return APICompletion(error: "No model loaded")
        }
        await apiGate.enter()
        let answer = await runCompletion(on: llamaContext, text: text, maxTokens: maxTokens, temperature: temperature)
        await apiGate.leave()
        return answer
    }

    private func runCompletion(on llamaContext: LlamaContext, text: String, maxTokens: Int, temperature: Float) async -> APICompletion {
        await llamaContext.set_temperature(temperature)
        if let why = await llamaContext.completion_init(text: text, max_tokens: Int32(maxTokens)) {
            await llamaContext.clear()
            return APICompletion(error: why)
        }
        var result = ""
        var tokenCount = 0

        while await !llamaContext.is_done && tokenCount < maxTokens {
            let piece = await llamaContext.completion_loop()
            result += piece
            tokenCount += 1
        }
        let promptTokens = Int(await llamaContext.n_prompt)
        let hitLimit = await llamaContext.hit_limit

        await llamaContext.clear()

        // Strip <think>...</think> blocks from Qwen-style reasoning models
        var cleaned = result
        if let thinkStart = cleaned.range(of: "<think>") {
            if let thinkEnd = cleaned.range(of: "</think>") {
                // Full think block — take everything after it
                cleaned = String(cleaned[thinkEnd.upperBound...])
            } else {
                // Think block consumed all tokens — take everything before it
                cleaned = String(cleaned[..<thinkStart.lowerBound])
            }
        }

        // Stop at first <|im_end|> — model may echo the prompt in a loop
        if let endTag = cleaned.range(of: "<|im_end|>") {
            cleaned = String(cleaned[..<endTag.lowerBound])
        }

        let cut = hitLimit || tokenCount >= maxTokens
        return APICompletion(text: cleaned.trimmingCharacters(in: .whitespacesAndNewlines), promptTokens: promptTokens, completionTokens: tokenCount, finishReason: cut ? "length" : "stop")
    }
}

/// A one-at-a-time gate for API completions: `enter` waits while another is running.
actor APIGate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    func enter() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func leave() {
        if waiting.isEmpty {
            busy = false
        } else {
            waiting.removeFirst().resume()
        }
    }
}
