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

import static jcuda.driver.JCudaDriver.*;

import java.util.*;
import java.util.concurrent.*;

import jcuda.Pointer;
import jcuda.driver.*;

/**
 * Defines the thread that handles the interop for the selected device.
 * 
 * @author Mehran Maghoumi
 *
 */
public class DeviceThread extends Thread {
	
	/** The capacity of the jobs queue */
	public static final int JOB_CAPACITY = 1;
	
	/** The LoadBalancer instance that has instantiated this object */
	protected TransScale parent = null;
	
	/** The device that this thread interops with */
	protected CUdevice device = null;
	
	/** The ordinal of the device that this thread interops with */
	protected int deviceNumber = -1;
	
	/** Invokables indexed by their ID */
	protected Map<String, Invokable> invokables = new HashMap<>();
	
	/** The queue that manages the module load requests */
	protected BlockingQueue<Kernel> kernelJobsQueue = new ArrayBlockingQueue<>(JOB_CAPACITY);
	
	/** Queue for all the kernel calls that this thread should make on the CUDA device */
	protected BlockingQueue<KernelInvoke> invocationJobs = new ArrayBlockingQueue<>(JOB_CAPACITY);
	
	/** Barrier to inform that something is available */
	protected Object barrier = new Object();
	
	/** The CUDA context of the current thread */ 
	protected CUcontext context = null;
	
	
	/**
	 * Initializes a thread that interops with the specified device
	 * @param deviceNumber	The ordinal of the device that this thread should interop with
	 */
	public DeviceThread(TransScale parent, int deviceNumber) {
		this.parent = parent;
		this.deviceNumber = deviceNumber;
		this.device = new CUdevice();
		setDaemon(true);
		cuDeviceGet(device, deviceNumber);
	}
	
	/**
	 * Add a new kernel to the list of kernels that this thread can invoke
	 * @param job
	 */
	public synchronized void addKernel(Kernel job) {
		
		synchronized(barrier) {
			synchronized(kernelJobsQueue) {
				try {
					kernelJobsQueue.put(job);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			interrupt();
		}
	}
	
	/**
	 * Queue a kernel invocation job on this device
	 * @param job
	 */
	public synchronized void queueJob(KernelInvoke job) {
		
		synchronized(barrier) {
			synchronized(invocationJobs) {
				try {
					invocationJobs.put(job);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			interrupt();
		}
	}
	
	
	/**
	 * Creates a new CUDA context and loads the supplied PTX module and obtains
	 * function handles to all the kernels that should be later invoked.
	 * 
	 * @param job
	 */
	protected void loadKernel(Kernel job) {
		// Notify parent that I am busy
		parent.notifyBusy(this);
		
		// Create and load the module for the new kernel
		CUmodule newModule = new CUmodule();		
		cuModuleLoad(newModule, job.ptxFile.getPath());
		
		// Store and index all function handlers
		for (String id : job.functionMapping.keySet()) {
			CUfunction newFunction = new CUfunction();			
			cuModuleGetFunction(newFunction, newModule, job.functionMapping.get(id));
			
			invokables.put(id, new Invokable(newModule, newFunction));
		}
		
		// Notify parent that I am available
		parent.notifyAvailable(this);
	}
	
	/** 
	 * A helper function to invoke a kernel using the specified job
	 * object.
	 * @param job
	 */
	private void invoke(KernelInvoke job) {
		// Notify parent that I am busy
		parent.notifyBusy(this);
		
		Invokable invokable = invokables.get(job.functionId);
		CUmodule module = invokable.module;
		CUfunction function = invokable.function;
		
		job.preTrigger.doTask(module);
		
		Pointer pointerToArgs = job.argSetter.getArgs();
		
		cuLaunchKernel(function, job.gridDimX, job.gridDimY, job.gridDimZ,
				job.blockDimX, job.blockDimY, job.blockDimZ,
				job.sharedMemorySize, null,
				pointerToArgs, null);
		cuCtxSynchronize();
		
		job.postTrigger.doTask(module);
		job.notifyComplete();
		
		// Notify parent that I am available
		parent.notifyAvailable(this);
	}	
	
	@Override
	public void run() {
		// Create a context for this device
		// NOTE: Context creation must be done on this thread!
		context = new CUcontext();
		cuCtxCreate(context, 0, device);
		cuCtxSetCurrent(context);
		cuCtxSetCurrent(context);
		
		while (true) {
			
			// Wait for something to become availble
			synchronized(barrier) {
				try {
					barrier.wait();
				} catch (InterruptedException e) {
				}
			}
			
			synchronized(kernelJobsQueue) {
				if (!kernelJobsQueue.isEmpty())
					loadKernel(kernelJobsQueue.poll());
			}
			
			synchronized(invocationJobs) {
				if (!invocationJobs.isEmpty())
					invoke(invocationJobs.poll());
			}
		}
		
	}	
}
