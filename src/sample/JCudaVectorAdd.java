package sample;


import java.io.*;
import java.util.HashMap;

import com.sir_m2x.transscale.*;
import com.sir_m2x.transscale.pointers.*;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

/**
 * This is a sample class demonstrating how to use the JCuda driver
 * bindings to load and execute a CUDA vector addition kernel.
 * The sample reads a CUDA file, compiles it to a PTX file
 * using NVCC, loads the PTX file as a module and executes
 * the kernel function. <br />
 */
public class JCudaVectorAdd
{
    /**
     * Entry point of this sample
     *
     * @param args Not used
     * @throws IOException If an IO error occurs
     */
    public static void main(String args[]) throws IOException
    {
    	JCuda.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		CudaPrimitive2D.usePitchedMemory(false);	// No need to automatically pitch memories for this sample 

		// Prepare the PTX file containing the kernel
		String ptxFileName = "";

		try {
			ptxFileName = TransScale.preparePtxFile("bin/sample/VectorAddKernel.cu", true, false);
		} catch (IOException e) {
			System.err.println("Could not create PTX file");
			throw new RuntimeException("Could not create PTX file", e);
		}
		
		// Create a kernel job and add it to TransScale, it will be loaded on all devices
		Kernel job = new Kernel();
		job.ptxFile = new File(ptxFileName);
		job.functionMapping = new HashMap<>();
		// Specify the kernel function name and unique identifier (used for distinguishing multiple kernels from different CUmodules
		job.functionMapping.put("add", "add");
		TransScale.getInstance().addKernel(job);
		
		final int numElements = 100000;

        // Allocate and fill the host input data
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        float hostOutput[] = new float[numElements];
        
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = (float)i;
            hostInputB[i] = (float)i;
        }
        
        
        // Initiate device arrays of type Float
        // The allocated arrays are 1D arrays, therefore their height is set to 1
        final CudaFloat2D deviceInputA = new CudaFloat2D(numElements, 1, 1, hostInputA, true);         
        final CudaFloat2D deviceInputB = new CudaFloat2D(numElements, 1, 1, hostInputB, true);
        final CudaFloat2D deviceOutput = new CudaFloat2D(numElements, 1, 1, hostOutput, true);
        
        // Define a sequence of operations that must be performed prior to executing the kernel
        // Usually desired memory spaces are allocated by a call to the reallocate() function
        Trigger pre = new Trigger() {
			@Override
			public void doTask(CUmodule module) {
				deviceInputA.reallocate();
				deviceInputB.reallocate();
				deviceOutput.reallocate();
			}
		};
		
		// Define the list of arguments that the kernel expects
		// Note that the overridden function must return a Pointer to a pointer of arguments
		KernelArgSetter setter = new KernelArgSetter() {
			
			@Override
			public Pointer getArgs() {
				return Pointer.to(Pointer.to(new int[] {numElements}), deviceInputA.toPointer(), deviceInputB.toPointer(), deviceOutput.toPointer());
			}
		};
        
		// Define a sequence of operations that must be performed after the execution of the kernel
        // Usually the output device pointers are refreshed (their GPU counterpart values are obtained)
		// and the allocated spaces are freed
		Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				// Fetch the calculated results
				deviceOutput.refresh();
				
				// Do clean up
				deviceInputA.free();
				deviceInputB.free();
				deviceOutput.free();
			}
		};
		
		// Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)numElements / blockSizeX);
		
		// Specify the kernel invocation parameters
        KernelInvoke kernelJob = new KernelInvoke();
		kernelJob.functionId = "add";
		kernelJob.preTrigger = pre;
		kernelJob.postTrigger = post;
		
		kernelJob.gridDimX = gridSizeX;
		kernelJob.gridDimY = 1;
		
		kernelJob.blockDimX = blockSizeX;
		kernelJob.blockDimY = 1;
		kernelJob.blockDimZ = 1;
		
		kernelJob.argSetter = setter;
		
		// Queue kernel and wait for it		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();	// Will block the calling thread until the device has finished processing
		
		hostOutput = deviceOutput.getArray();

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numElements; i++)
        {
            float expected = i+i;
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));
    }
}