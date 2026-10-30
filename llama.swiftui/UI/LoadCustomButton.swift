import SwiftUI
import UniformTypeIdentifiers

struct LoadCustomButton: View {
    @ObservedObject private var llamaState: LlamaState
    @State private var showFileImporter = false

    init(llamaState: LlamaState) {
        self.llamaState = llamaState
    }

    var body: some View {
        VStack {
            Button(action: {
                showFileImporter = true
            }) {
                Text("Load a model from Files")
            }
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [UTType(filenameExtension: "gguf", conformingTo: .data)!],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let files):
                // Copied into the app's own folder, so it stays in Downloaded Models and still opens after the app restarts.
                files.forEach { file in
                    let gotAccess = file.startAccessingSecurityScopedResource()
                    if !gotAccess { return }
                    let target = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(file.lastPathComponent)
                    llamaState.messageLog += "Copying \(file.lastPathComponent) into the app...\n"
                    Task.detached {
                        var failure: String? = nil
                        if !FileManager.default.fileExists(atPath: target.path()) {
                            do {
                                try FileManager.default.copyItem(at: file, to: target)
                            } catch {
                                failure = error.localizedDescription
                            }
                        }
                        file.stopAccessingSecurityScopedResource()
                        await MainActor.run {
                            if let failure {
                                llamaState.messageLog += "Could not copy it: \(failure)\n"
                                return
                            }
                            if !llamaState.downloadedModels.contains(where: { $0.filename == target.lastPathComponent }) {
                                llamaState.downloadedModels.append(Model(name: target.deletingPathExtension().lastPathComponent, url: "", filename: target.lastPathComponent, status: "downloaded"))
                            }
                            do {
                                try llamaState.loadModel(modelUrl: target)
                            } catch {
                                llamaState.messageLog += "Could not load it: \(error.localizedDescription)\n"
                            }
                        }
                    }
                }
            case .failure(let error):
                print(error)
            }
        }
    }
}
