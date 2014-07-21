/*
 * Copyright 2014 Mehran Maghoumi
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.sir_m2x.transscale;

/**
 * Defines the elements that are required for invoking a kernel in CUDA.
 * This class defines the operations that should be done before and after 
 * a CUDA kernel call
 * 
 * @author Mehran Maghoumi
 *
 */
public class KernelInvoke {
	
	/** The ID of the kernel to call */
	public String functionId;
	
	/** Pre-call triggers */
	public Trigger preTrigger;
	
	/** Post-call triggers */
	public Trigger postTrigger;
	
	/** A function that returns the pointer to the kernel arguments */
	public KernelArgSetter argSetter;
	
	/** Grid size in X */
	public int gridDimX;
	
	/** Grid size in Y */
	public int gridDimY;
	
	/** Grid size in Z */
	public int gridDimZ = 1;
	
	/** Block size in X */
	public int blockDimX;
	
	/** Block size in Y */
	public int blockDimY;
	
	/** Block size in Z */
	public int blockDimZ;
	
	/** The size of the shared memory to pass to the kernel function */
	public int sharedMemorySize = 0;
	
	/** Can be used for debugging purposes */
	public String id;
	
	/** Flag indicating that this job has completed */
	protected volatile boolean jobComplete = false;
	
	/** For waiting for this job to complete */
	protected Object mutex = new Object();
	
	/** Holds a reference to the thread that is currently waiting for this job to conclude */
	protected Thread waitingThread = null;
	
	/**
	 * @return	True if this job is complete, false otherwise
	 */
	public boolean isComplete() {
		return this.jobComplete;
	}
	
	/**
	 * Wait for this job to complete. Blocks the calling thread until
	 * this job has been completed on the graphics card.
	 */
	public final void waitFor() {
		
		synchronized(mutex) {
			while (!jobComplete) {
				try {
					waitingThread = Thread.currentThread();
					mutex.wait();
				} catch (InterruptedException e) {}
			}
		}
	}
	
	/**
	 * Notify the threads that are waiting for this job to finish
	 */
	public final void notifyComplete() {
		
		synchronized (mutex) {
			jobComplete = true;
			
			if (waitingThread != null)
				waitingThread.interrupt();			
		}
	}
}
