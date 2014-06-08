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

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

/**
 * Defines the elements that are required to make a kernel function
 * invokable in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class Invokable { 
	
	/** The module that this function is being invoked on */
	public CUmodule module;
	
	/** The CUDA kernel that could be invoked */
	public CUfunction function;
	
	public Invokable(CUmodule module, CUfunction function) {
		this.module = module;
		this.function = function;
	}
}
