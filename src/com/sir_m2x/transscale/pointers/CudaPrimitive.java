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

package com.sir_m2x.transscale.pointers;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

/**
 * An extension of CUdeviceptr that represents a primitive variable in CUDA with
 * an equivalent variable in Java. The values are synched between the GPU and the
 * CPU.  
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class CudaPrimitive extends CUdeviceptr implements Cloneable {
	
	/**
	 * A flag indicating that this pointer has been freed and subsequent operations
	 * cannot be performed unless this pointer has been reallocated on the GPU memory
	 */
	protected boolean freed = false;
	
	/**
	 * @return	The size of this pointer in bytes
	 */
	public abstract int getSizeInBytes();
	
	/**
	 * Fetch the current value of this pointer from the GPU memory
	 * @return	The error code returned by JCuda
	 */
	public abstract int refresh();
	
	/** Reallocates this pointer if it was freed */
	public abstract int reallocate();
	
	/**
	 * @return	True if this pointer is freed from the GPU memory,
	 * 			False otherwise
	 */
	public boolean isFreed() {
		return this.isFreed();
	}
	
	/**
	 * Frees the allocated memory but keeps the host data intact
	 * WARNING: Do not use cuMemFree on this object directly!
	 */
	public void free() {
		if (freed)
			throw new RuntimeException("Already freed");
		if (!freed)
			cuMemFree(this);
		
		freed = true;
	}
	
	/**
	 * Creates a pointer to the GPU pointer of this object
	 * @return The native pointer to pointer to GPU memory space
	 */
	public Pointer toPointer() {
		return Pointer.to(this);
	}
	
	/**
	 * Convert's the underlying host data of this object to a Pointer
	 */
	public abstract Pointer hostDataToPointer();
	
	@Override
	public abstract Object clone();
	
}
