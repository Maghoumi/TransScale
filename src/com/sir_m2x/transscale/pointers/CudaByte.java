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
import jcuda.Sizeof;
import static jcuda.driver.JCudaDriver.*;

/**
 * Represents a primitive byte that is synchronized with a char pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaByte extends CudaPrimitive {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected byte byteValue = 0;
	
	public CudaByte() {
		this((byte) 0);
	}
	
	public CudaByte (byte initialValue) {
		this.byteValue = initialValue;
		cuMemAlloc(this, getSizeInBytes());
	}
	
	/**
	 * @return	The cached value of the variable pointed by this pointer.
	 * 			Again, note that this is a cached value!
	 */
	public byte getValue() {
		return this.byteValue;
	}
	
	/**
	 * Set the value of this pointer to a new value
	 * @param newValue	The new value to be represented by the memory space of this pointer
	 * @return	JCuda's error code
	 */
	public int setValue(byte newValue) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		this.byteValue = newValue;
		return cuMemcpyHtoD(this, Pointer.to(new byte[] {byteValue}), getSizeInBytes());
	}

	@Override
	public int getSizeInBytes() {
		return Sizeof.BYTE;
	}

	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		byte[] newValue = new byte[1];
		int errCode = cuMemcpyDtoH(Pointer.to(newValue), this, getSizeInBytes());  
		this.byteValue = newValue[0];
		return errCode;
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		cuMemAlloc(this, getSizeInBytes());
		return setValue(this.byteValue);
	}

	@Override
	public Object clone() {
		return new CudaInteger(byteValue);
	}

	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(new byte[] {this.byteValue});
	}
}
