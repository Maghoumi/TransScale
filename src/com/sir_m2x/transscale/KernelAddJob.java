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

import java.io.File;
import java.util.Map;

/**
 * Defines the elements that are required to load a kernel into all devices.
 * 
 * @author Mehran Maghoumi
 *
 */
public class KernelAddJob {
	
	/** The PTX file containing the CUDA code that we want to load */
	public File ptxFile;
	
	/**
	 * To solve name collision problem among different kernels in different PTX files, I 
	 * decided to make function calls possible using their ID. If there are two kernels
	 * both named "add", to discern which one should be called, we assign them different IDs.
	 * The thread that wants to call the function will resolve this collision.
	 * This map does the following:
	 * 	Add ==> Add
	 * 	Add2 ==> Add (later in another PTX file)
	 */
	public Map<String, String> functionMapping;
	
}
