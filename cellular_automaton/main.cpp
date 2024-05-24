// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
//#include <gl/GL.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "Header.cuh"
#include "matrix.cuh"
#include "tests.cuh"
#include <cuda_gl_interop.h>
//#include <cuComplex.h>


#include "cufft.h"

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}


void compute_kernel(Matrix convKernel, float* shiftedKernel,
                    float2* kernelSpectrum, float* kernel_sum,
                    KernelParams params, cufftHandle fftPlanFwd) {
    //getRing(convKernel, params.width/2.0f, params.height/2.0f, params.ring_radiuses[0], params.ring_sigmas[0]);
    launchRings(convKernel, params.width / 2.0f, params.height / 2.0f, params);
    *kernel_sum = launch_reduceSum(convKernel.device_data, params.width*params.height);
	launch_fftShift2d(shiftedKernel, convKernel);
    cufftExecR2C(fftPlanFwd, (cufftReal*)shiftedKernel, (cufftComplex*)kernelSpectrum);
}

// Main code
int main(int, char**)
{
    test_sum();
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1024, 1024, "Cellular Automaton", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync. Set to 0 for UNLIMITED SPEED

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    const int image_width = 2048;//1024;
    const int image_height = 2048;//1024;
    //create texture
    GLuint tex_cudaResult;
    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); //_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP); //_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, NULL);
    //SDK_CHECK_ERROR_GL();

    //Declare cudaGraphicsResource, register it to texture
    struct cudaGraphicsResource* cuda_tex_result_resource;
    cudaGraphicsGLRegisterImage(
        &cuda_tex_result_resource, tex_cudaResult, GL_TEXTURE_2D,
        cudaGraphicsMapFlagsWriteDiscard);

    //initialise cuda pointer with pixel data
    unsigned int* cuda_dest_resource;
    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cudaMalloc((void**)&cuda_dest_resource, size_tex_data);

    int frame_num = 0;
    float speed = 1.;

    Matrix state;
    state.width = image_width;
    state.height = image_height;
    cudaMalloc((void**)&state.device_data, state.width * state.height * sizeof(float));
    //getGaussianBlob(state, 200, 200, 50);
    getSquare(state, 300, 300, 600, 600, 1.0f, 0.f);
    //getRing(state, 22.5f, 22.5f, 15, 3.75);

    Matrix convKernel;
    convKernel.width = image_width;
    convKernel.height = image_height;
    cudaMalloc((void**)&convKernel.device_data, convKernel.width * convKernel.height * sizeof(float));
    KernelParams kernelParams;
    kernelParams.n_rings = 3;
    kernelParams.ring_radiuses = (float*)malloc(kernelParams.n_rings * sizeof(float));
    //kernelParams.ring_sigma = 8.0f;
    kernelParams.ring_radiuses[0] = 30.0f;
    kernelParams.ring_radiuses[1] = 70.0f;
    kernelParams.ring_radiuses[2] = 110.0f;
    kernelParams.ring_sigmas = (float*)malloc(kernelParams.n_rings * sizeof(float));
    kernelParams.ring_sigmas[0] = 8.0f;
    kernelParams.ring_sigmas[1] = 8.0f;
    kernelParams.ring_sigmas[2] = 8.0f;
    kernelParams.ring_coefs = (float*)malloc(kernelParams.n_rings * sizeof(float));
    kernelParams.ring_coefs[0] = 1.0f;
    kernelParams.ring_coefs[1] = 0.5f;
    kernelParams.ring_coefs[2] = 0.2f;
    kernelParams.width = image_width;
    kernelParams.height = image_height;

    float kernel_sum = 0.0f;
    //printf("KERNEL SUM: %f", kernel_sum);
    float* shiftedKernel;
    cudaMalloc((void**)&shiftedKernel, convKernel.width * convKernel.height * sizeof(float));

    //prepare spectrum pointers
    float2* kernelSpectrum;
    cudaMalloc((void**)&kernelSpectrum, image_width * (image_height/2+1) * sizeof(float2));
    float2* dataSpectrum;
    cudaMalloc((void**)&dataSpectrum, image_width * (image_height/2+1) * sizeof(float2));
    float2* resultSpectrum;
    cudaMalloc((void**)&resultSpectrum, image_width * (image_height/2+1) * sizeof(float2));
    float* resultConv;
    cudaMalloc((void**)&resultConv, image_width * image_height * sizeof(float));


    cufftHandle fftPlanFwd, fftPlanInv;

    cufftPlan2d(&fftPlanFwd, image_height, image_width, CUFFT_R2C);
    cufftPlan2d(&fftPlanInv, image_height, image_width, CUFFT_C2R);

    compute_kernel(convKernel, shiftedKernel, kernelSpectrum, &kernel_sum, kernelParams, fftPlanFwd);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching C2R!\n", cudaStatus);
    }

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //ImGui::ShowDemoWindow(&show_demo_window);
        bool recompute = false;

        {
            static float f = 0.0f;

            ImGui::Begin("Params");                          // Create a window called "Hello, world!" and append into it.

            ImGui::SliderFloat("Speed", &speed, 0.1f, 10.0f);

            for (int ringIndex=0; ringIndex < kernelParams.n_rings; ringIndex++) {
                if (true) {
                    char label_radius[] = "Kernel radius #0";
                    sprintf(label_radius, "Kernel radius #%d", ringIndex);
                    recompute = ImGui::SliderFloat(label_radius, &kernelParams.ring_radiuses[ringIndex], 15.0f, 100.0f) || recompute;
                    char label_sigma[] = "Ring sigma #0";
                    sprintf(label_sigma, "Ring sigma #%d", ringIndex);
                    recompute = ImGui::SliderFloat(label_sigma, &kernelParams.ring_sigmas[ringIndex], 3.0f, 15.0f) || recompute;
                }
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        if (recompute) {
            compute_kernel(convKernel, shiftedKernel, kernelSpectrum, &kernel_sum, kernelParams, fftPlanFwd);
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        //update with frame
		cufftExecR2C(fftPlanFwd, (cufftReal*)state.device_data, (cufftComplex*)dataSpectrum);

		launch_multiplyComplex(resultSpectrum, kernelSpectrum, dataSpectrum,
							   image_width * (image_height / 2 + 1), image_width*image_height*kernel_sum);
        cufftExecC2R(fftPlanInv, (cufftComplex*)resultSpectrum, (cufftReal*)resultConv); //state.device_data);
        launch_updateState(state, resultConv, .5, .15, 1.f/io.Framerate, speed);

	    launch_toRGB(cuda_dest_resource, state);
        //DEBUG
        //convKernel.device_data = shiftedKernel;
        //launch_toRGB(cuda_dest_resource, convKernel);

        frame_num++;
        //Transfer cuda computed pixel in texture
        cudaArray* texture_ptr;
        cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
        cudaGraphicsSubResourceGetMappedArray(
            &texture_ptr, cuda_tex_result_resource, 0, 0);
        
        cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource,
            size_tex_data, cudaMemcpyDeviceToDevice);

        cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
        
        //draw texture full window
        glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 0.0);
        glVertex3f(1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 1.0);
        glVertex3f(1.0, 1.0, 0.5);
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-1.0, 1.0, 0.5);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaFree(cuda_dest_resource);
    cudaFree(state.device_data);
    cudaFree(convKernel.device_data);
    cudaFree(shiftedKernel);
    cudaFree(dataSpectrum);
    cudaFree(resultSpectrum);
    cudaFree(resultConv);

    return 0;
}
