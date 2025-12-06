import * as FileSystem from 'expo-file-system/legacy';

interface AnalysisParams {
  image: string;
  overlay_size: number;
  diameter: number;
}

interface ARValueParams {
  ribAngle: number;
  ribHeight: number;
  ribLength: number;
  numRows: number;
}

interface ApiResponse {
  status: string;
  data?: any;
  error?: string;
  [key: string]: any;
}

export class ApiService {
  // Try localhost first, then fallback to the correct IP
  private static readonly baseUrl = 'http://192.168.29.144:5000'; 
  private static readonly fallbackUrl = 'http://127.0.0.1:5000'; 
  private static readonly timeout = 180000; // 180 seconds (3 minutes) for complex operations like SAM segmentation
  private static readonly serverCheckTimeout = 5000; // 5 seconds
  
  // Cache for API responses
  private static responseCache: Map<string, ApiResponse> = new Map();
  private static cacheTimestamps: Map<string, number> = new Map();
  private static readonly cacheDuration = 5 * 60 * 1000; // 5 minutes

  // Check if server is available
  static async isServerAvailable(): Promise<boolean> {
    let retryCount = 0;
    const maxRetries = 3;
    
    // Try multiple candidate URLs (device, emulator, localhost, Metro host)
    const candidateUrls = [
      this.baseUrl,
      this.fallbackUrl,
      'http://localhost:5000',
      'http://10.0.2.2:5000', // Android emulator to host
      'http://10.9.76.163:5000', // Metro LAN IP seen in logs
      'http://192.168.150.222:5000', // Previously used LAN IP
    ];
    // De-duplicate and drop falsy
    const urls = Array.from(new Set(candidateUrls.filter(Boolean)));
    
    for (const url of urls) {
      while (retryCount < maxRetries) {
        try {
          console.log(`Checking server availability at ${url} (attempt ${retryCount + 1}/${maxRetries})...`);
          
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), this.serverCheckTimeout);
          
          const response = await fetch(`${url}/check_server`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
            signal: controller.signal,
          });
          
          clearTimeout(timeoutId);
          
          console.log('Server response status:', response.status);
          if (response.status === 200) {
            console.log('Server is available at:', url);
            // Update baseUrl to the working URL
            (this as any).baseUrl = url;
            return true;
          }
          
          console.log('Server returned non-200 status:', response.status);
          retryCount++;
          if (retryCount < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        } catch (error) {
          console.log('Error checking server at', url, ':', error);
          retryCount++;
          if (retryCount < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      retryCount = 0; // Reset for next URL
    }
    
    console.log('Server check failed after trying all URLs');
    return false;
  }

  // Ring Test: process cross-section image (multipart/form-data)
  static async processRingTest({
    imageBase64,
    diameter,
  }: {
    imageBase64: string;
    diameter: number;
  }): Promise<ApiResponse> {
    // Ensure server is up
    if (!await this.isServerAvailable()) {
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }

    // Write base64 to a temporary file so we can send as file upload
    const fileName = `ring_test_${Date.now()}.jpg`;
    // Prefer cache directory; Expo provides file:// URIs
    const fileUri = `${FileSystem.cacheDirectory}${fileName}`;
    await FileSystem.writeAsStringAsync(fileUri, imageBase64.replace(/^data:image\/[a-zA-Z]+;base64,/, ''), { encoding: 'base64' });

    const form = new FormData();
    // React Native fetch FormData file object shape
    form.append('image' as any, {
      uri: fileUri,
      name: fileName,
      type: 'image/jpeg',
    } as any);
    form.append('diameter' as any, String(diameter));

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const endpoint = `${this.baseUrl}/process-ring-test`;
      const res = await fetch(endpoint, {
        method: 'POST',
        // Let fetch set the correct multipart boundary; adding Accept for iOS
        headers: { Accept: 'application/json' } as any,
        body: form as any,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      // Cleanup temp file (best-effort)
      try { await FileSystem.deleteAsync(fileUri, { idempotent: true }); } catch {}

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Ring test failed (${res.status}): ${errText}`);
      }
      const json = await res.json();
      return json as ApiResponse;
    } finally {
      // Ensure we clear timeout if we threw
      clearTimeout(timeoutId);
    }
  }

  // Ring Test: fetch report by test_id
  static async getRingReport(testId: string): Promise<ApiResponse> {
    if (!await this.isServerAvailable()) {
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.serverCheckTimeout * 4);
    try {
      const res = await fetch(`${this.baseUrl}/get-ring-report?test_id=${encodeURIComponent(testId)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Get ring report failed (${res.status}): ${errText}`);
      }
      return await res.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // Helper method to get cache key
  private static getCacheKey(endpoint: string, params: any): string {
    return `${endpoint}:${JSON.stringify(params)}`;
  }

  // Helper method to check if cache is valid
  private static isCacheValid(key: string): boolean {
    if (!this.cacheTimestamps.has(key)) return false;
    const timestamp = this.cacheTimestamps.get(key)!;
    return Date.now() - timestamp < this.cacheDuration;
  }

  // Helper method to compress image before sending
  private static async compressImage(base64Image: string): Promise<string> {
    if (base64Image.length < 100000) return base64Image; // Skip if already small
    
    // For now, just return the original image
    // In a real implementation, you might want to use a library like react-native-image-resizer
    return base64Image;
  }

  // Analyze angle and length
  static async analyzeAngleAndLength({
    imageBase64,
    diameter,
    overlaySize,
    scale_factor,
    calibration_method,
  }: {
    imageBase64: string;
    diameter: number;
    overlaySize?: number;
    scale_factor?: number;
    calibration_method?: string;
  }): Promise<ApiResponse | null> {
    console.log('Starting analyzeAngleAndLength...');
    
    // Check server availability first
    if (!await this.isServerAvailable()) {
      console.log('Server not available');
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }

    const params: any = {
      image: await this.compressImage(imageBase64),
      overlay_size: overlaySize,
      diameter: diameter,
    };
    
    // Add calibration data if available
    if (scale_factor !== undefined) {
      params.scale_factor = scale_factor;
      console.log('Using calibrated scale factor:', scale_factor);
    }
    
    if (calibration_method !== undefined) {
      params.calibration_method = calibration_method;
      console.log('Using calibration method:', calibration_method);
    }
    
    const cacheKey = this.getCacheKey('analyze_angle_and_length', params);
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log('Returning cached result for analyzeAngleAndLength');
      return this.responseCache.get(cacheKey) || null;
    }

    try {
      console.log('Making API request to analyze_angle_and_length...');
      const url = `${this.baseUrl}/analyze_angle_and_length?diameter=${diameter}`;
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Successfully parsed response:', result);
        // Cache the result
        this.responseCache.set(cacheKey, result);
        this.cacheTimestamps.set(cacheKey, Date.now());
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in analyzeAngleAndLength:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timed out. The server is taking too long to respond. Please try again.');
      } else if (error instanceof Error && error.message.includes('NetworkError')) {
        throw new Error('Could not connect to the server. Please check if the server is running and try again.');
      }
      throw error;
    }
  }

  // Analyze rib height
  static async analyzeRibHeight({
    imageBase64,
    diameter,
    overlaySize,
    scale_factor,
    calibration_method,
  }: {
    imageBase64: string;
    diameter: number;
    overlaySize?: number;
    scale_factor?: number;
    calibration_method?: string;
  }): Promise<ApiResponse | null> {
    console.log('Starting analyzeRibHeight...');
    
    // Check server availability first
    if (!await this.isServerAvailable()) {
      console.log('Server not available');
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }
    
    const params: any = {
      image: await this.compressImage(imageBase64),
      diameter: diameter,
    };
    
    // Only include overlay_size if provided
    if (overlaySize !== undefined) {
      params.overlay_size = overlaySize;
    }
    
    // Include calibration data if provided
    if (scale_factor !== undefined) {
      params.scale_factor = scale_factor;
    }
    
    if (calibration_method !== undefined) {
      params.calibration_method = calibration_method;
    }
    
    const cacheKey = this.getCacheKey('analyze_rib_height', params);
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log('Returning cached result for analyzeRibHeight');
      return this.responseCache.get(cacheKey) || null;
    }

    try {
      console.log('Making API request to analyze_rib_height...');
      const url = `${this.baseUrl}/analyze_rib_height?diameter=${diameter}`;
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      // Height analysis with SAM calibration needs more time - use extended timeout
      const extendedTimeout = 180000; // 3 minutes for height analysis
      const timeoutId = setTimeout(() => controller.abort(), extendedTimeout);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Successfully parsed response:', result);
        // Cache the result
        this.responseCache.set(cacheKey, result);
        this.cacheTimestamps.set(cacheKey, Date.now());
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in analyzeRibHeight:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timed out. The server is taking too long to respond. Please try again.');
      } else if (error instanceof Error && error.message.includes('NetworkError')) {
        throw new Error('Could not connect to the server. Please check if the server is running and try again.');
      }
      throw error;
    }
  }

  // Test quick detection endpoint
  static async testQuickEndpoint(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/test_quick`, {
        method: 'GET',
        headers: {'Content-Type': 'application/json'},
      });
      
      if (response.ok) {
        console.log('Quick detection endpoint test: SUCCESS');
        return true;
      } else {
        console.log('Quick detection endpoint test: FAILED');
        return false;
      }
    } catch (error) {
        console.log('Quick detection endpoint test: ERROR', error);
        return false;
    }
  }

  // Test simple POST endpoint with image data
  static async testSimplePost(): Promise<boolean> {
    try {
      // Create a small test image (1x1 pixel)
      const testImageData = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
      
      const response = await fetch(`${this.baseUrl}/test_quick_simple`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          image: testImageData,
          diameter: 10
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Simple POST test: SUCCESS', result);
        return true;
      } else {
        const errorText = await response.text();
        console.log('Simple POST test: FAILED', errorText);
        return false;
      }
    } catch (error) {
      console.log('Simple POST test: ERROR', error);
      return false;
    }
  }

  // Quick YOLO detection only (fast, no SAM)
  static async quickDetect({
    imageBase64,
    diameter,
  }: {
    imageBase64: string;
    diameter: number;
  }): Promise<ApiResponse | null> {
    console.log('Starting quickDetect...');
    
    // Check server availability first
    if (!await this.isServerAvailable()) {
      console.log('Server not available');
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }
    
    // Compress image for quick detection
    const compressedImage = await this.compressImage(imageBase64);
    console.log(`Image compressed from ${imageBase64.length} to ${compressedImage.length} characters`);
    
    const params = {
      image: compressedImage,
      diameter: diameter,
    };
    
    try {
      console.log('Making API request to quick_detect...');
      const url = `${this.baseUrl}/quick_detect?diameter=${diameter}`;
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout (1 minute) for large images
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Quick detection result:', result);
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in quickDetect:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Quick detection timed out. Please try again.');
      }
      throw error;
    }
  }

  // Detect TMT bar in image
  static async detectTmtBar({
    imageBase64,
    diameter,
    overlaySize,
    tmt_bounding_box,
  }: {
    imageBase64: string;
    diameter: number;
    overlaySize?: number;
    tmt_bounding_box?: [number, number, number, number];
  }): Promise<ApiResponse | null> {
    console.log('Starting detectTmtBar...');
    
    // Check server availability first
    if (!await this.isServerAvailable()) {
      console.log('Server not available');
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }
    
    const params: any = {
      image: await this.compressImage(imageBase64),
      diameter: diameter,
    };
    
    // Only include overlay_size if provided
    if (overlaySize !== undefined) {
      params.overlay_size = overlaySize;
    }
    
    // Only include tmt_bounding_box if provided
    if (tmt_bounding_box !== undefined) {
      params.tmt_bounding_box = tmt_bounding_box;
    }
    
    const cacheKey = this.getCacheKey('detect_tmt_bar', params);
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log('Returning cached result for detectTmtBar');
      return this.responseCache.get(cacheKey) || null;
    }

    try {
      console.log('Making API request to detect_tmt_bar...');
      const url = `${this.baseUrl}/detect_tmt_bar?diameter=${diameter}`;
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      // Extended timeout for SAM segmentation (can take 60+ seconds)
      const extendedTimeout = 180000; // 3 minutes for SAM segmentation
      const timeoutId = setTimeout(() => controller.abort(), extendedTimeout);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Successfully parsed response:', result);
        // Cache the result
        this.responseCache.set(cacheKey, result);
        this.cacheTimestamps.set(cacheKey, Date.now());
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in detectTmtBar:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timed out. The server is taking too long to respond. Please try again.');
      } else if (error instanceof Error && error.message.includes('NetworkError')) {
        throw new Error('Could not connect to the server. Please check if the server is running and try again.');
      }
      throw error;
    }
  }

  // Unified rib analysis - analyzes all rib parameters from a single image
  static async analyzeRibUnified({
    imageBase64,
    segmentedImageBase64,
    diameter,
    overlaySize,
    scale_factor,
    calibration_method,
    brand,
  }: {
    imageBase64: string;
    segmentedImageBase64?: string;
    diameter: number;
    overlaySize?: number;
    scale_factor?: number;
    calibration_method?: string;
    brand?: string;
  }): Promise<ApiResponse | null> {
    console.log('Starting analyzeRibUnified...');
    
    // Check server availability first
    if (!await this.isServerAvailable()) {
      console.log('Server not available');
      throw new Error('Server is not available. Please ensure the backend server is running.');
    }

    const params: any = {
      image: await this.compressImage(imageBase64),
      diameter: diameter,
    };
    
    // Add segmented image if provided (OPTIMIZATION: avoids duplicate SAM segmentation)
    if (segmentedImageBase64) {
      params.segmented_image = await this.compressImage(segmentedImageBase64);
      console.log('✅ OPTIMIZATION: Using pre-segmented image - backend will skip SAM segmentation');
    } else {
      console.log('⚠️  No pre-segmented image provided - backend will perform SAM segmentation');
    }
    
    // Only include overlay_size if provided
    if (overlaySize !== undefined) {
      params.overlay_size = overlaySize;
    }
    
    // Add calibration data if available
    if (scale_factor !== undefined) {
      params.scale_factor = scale_factor;
      console.log('Using calibrated scale factor:', scale_factor);
    }
    
    if (calibration_method !== undefined) {
      params.calibration_method = calibration_method;
      console.log('Using calibration method:', calibration_method);
    }
    
    const cacheKey = this.getCacheKey('analyze_rib_unified', params);
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log('Returning cached result for analyzeRibUnified');
      return this.responseCache.get(cacheKey) || null;
    }

    try {
      console.log('Making API request to analyze_rib_unified...');
      let url = `${this.baseUrl}/analyze_rib_unified?diameter=${diameter}`;
      if (brand) {
        url += `&brand=${brand}`;
      }
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      // Extended timeout for comprehensive analysis
      const extendedTimeout = 240000; // 4 minutes for unified analysis
      const timeoutId = setTimeout(() => controller.abort(), extendedTimeout);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Successfully parsed response:', result);
        // Cache the result
        this.responseCache.set(cacheKey, result);
        this.cacheTimestamps.set(cacheKey, Date.now());
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in analyzeRibUnified:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timed out. The unified analysis is taking too long to respond. Please try again.');
      } else if (error instanceof Error && error.message.includes('NetworkError')) {
        throw new Error('Could not connect to the server. Please check if the server is running and try again.');
      }
      throw error;
    }
  }

  // Calculate AR value
  static async calculateARValue({
    angle,
    height,
    length,
    numRows,
  }: {
    angle: number;
    height: number;
    length: number;
    numRows: number;
  }): Promise<ApiResponse | null> {
    console.log('Starting calculateARValue...');
    const params: ARValueParams = {
      ribAngle: angle,
      ribHeight: height,
      ribLength: length,
      numRows: numRows,
    };
    
    const cacheKey = this.getCacheKey('calculate_ar_from_params', params);
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log('Returning cached result for calculateARValue');
      return this.responseCache.get(cacheKey) || null;
    }

    try {
      console.log('Making API request to calculate_ar_from_params...');
      const url = `${this.baseUrl}/calculate_ar_from_params`;
      console.log('Request URL:', url);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      console.log('Received response with status code:', response.status);
      
      if (response.status === 200) {
        const result: ApiResponse = await response.json();
        console.log('Successfully parsed response:', result);
        // Cache the result
        this.responseCache.set(cacheKey, result);
        this.cacheTimestamps.set(cacheKey, Date.now());
        return result;
      } else {
        const errorBody = await response.text();
        console.log('Error response body:', errorBody);
        throw new Error(`Server returned status code ${response.status}: ${errorBody}`);
      }
    } catch (error) {
      console.log('Error in calculateARValue:', error);
      throw error;
    }
  }

  // Clear cache
  static clearCache(): void {
    this.responseCache.clear();
    this.cacheTimestamps.clear();
    console.log('Frontend cache cleared successfully');
  }

  // Clear backend cache (if endpoint exists)
  static async clearBackendCache(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/clear_cache`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
      });
      
      if (response.ok) {
        console.log('Backend cache cleared successfully');
      } else {
        console.log('Backend cache clear endpoint not available or failed');
      }
    } catch (error) {
      console.log('Backend cache clear not available:', error);
    }
  }
} 