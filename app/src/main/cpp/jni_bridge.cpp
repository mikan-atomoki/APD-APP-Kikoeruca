#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "apd_inference.h"
#include "apd_kernels.h"

static apd::Inference* g_inference = nullptr;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_kikoeruca_debug_APDEngine_nativeLoadModel(
    JNIEnv* env, jobject, jobject asset_manager, jstring path)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    const char* c_path = env->GetStringUTFChars(path, nullptr);
    AAsset* asset = AAssetManager_open(mgr, c_path, AASSET_MODE_BUFFER);
    env->ReleaseStringUTFChars(path, c_path);

    if (!asset) {
        LOGE("Failed to open asset");
        return 0;
    }

    const size_t size = AAsset_getLength(asset);
    const void* data = AAsset_getBuffer(asset);

    auto* inf = new apd::Inference();
    bool ok = inf->load(static_cast<const uint8_t*>(data), size);
    AAsset_close(asset);

    if (!ok) {
        delete inf;
        return 0;
    }

    g_inference = inf;
    return reinterpret_cast<jlong>(inf);
}

JNIEXPORT jfloatArray JNICALL
Java_com_kikoeruca_debug_APDEngine_nativeInfer(
    JNIEnv* env, jobject, jlong handle, jfloatArray audio)
{
    auto* inf = reinterpret_cast<apd::Inference*>(handle);
    if (!inf) return nullptr;

    jint len = env->GetArrayLength(audio);
    jfloat* data = env->GetFloatArrayElements(audio, nullptr);

    auto result = inf->run(data, len);

    env->ReleaseFloatArrayElements(audio, data, JNI_ABORT);

    // Return [pre_sigmoid, score, inference_ms]
    jfloatArray ret = env->NewFloatArray(3);
    jfloat buf[3] = { result.pre_sigmoid, result.score, result.inference_ms };
    env->SetFloatArrayRegion(ret, 0, 3, buf);
    return ret;
}

JNIEXPORT jfloat JNICALL
Java_com_kikoeruca_debug_APDEngine_nativeComputeRms(
    JNIEnv* env, jobject, jfloatArray audio)
{
    jint len = env->GetArrayLength(audio);
    jfloat* data = env->GetFloatArrayElements(audio, nullptr);
    float rms = apd::kernels::compute_rms(data, len);
    env->ReleaseFloatArrayElements(audio, data, JNI_ABORT);
    return rms;
}

JNIEXPORT void JNICALL
Java_com_kikoeruca_debug_APDEngine_nativeRelease(
    JNIEnv*, jobject, jlong handle)
{
    auto* inf = reinterpret_cast<apd::Inference*>(handle);
    if (inf) {
        delete inf;
        if (g_inference == inf) g_inference = nullptr;
    }
}

} // extern "C"
