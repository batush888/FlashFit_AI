import { api, API_ENDPOINTS } from './client';
import {
  UploadResponse,
  ApiResponse,
  UploadProgress,
} from '../types';

/**
 * 上传相关API服务
 */
export class UploadService {
  /**
   * 上传单个图片
   */
  async uploadImage(
    file: File,
    tags?: string[],
    onProgress?: (progress: UploadProgress) => void
  ): Promise<ApiResponse<UploadResponse>> {
    // 验证文件
    const validation = this.validateFile(file);
    if (!validation.isValid) {
      return {
        success: false,
        error: validation.error,
      };
    }

    // 创建FormData
    const formData = new FormData();
    formData.append('file', file);
    
    if (tags && tags.length > 0) {
      formData.append('tags', JSON.stringify(tags));
    }

    // 上传进度回调
    const progressCallback = (progress: number) => {
      if (onProgress) {
        onProgress({
          loaded: (progress / 100) * file.size,
          total: file.size,
          percentage: progress,
        });
      }
    };

    return await api.upload<UploadResponse>(
      API_ENDPOINTS.UPLOAD.IMAGE,
      formData,
      progressCallback
    );
  }

  /**
   * 批量上传图片
   */
  async uploadBatch(
    files: File[],
    onProgress?: (fileIndex: number, progress: UploadProgress) => void,
    onComplete?: (fileIndex: number, result: ApiResponse<UploadResponse>) => void
  ): Promise<ApiResponse<UploadResponse[]>> {
    const results: UploadResponse[] = [];
    const errors: string[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      try {
        const result = await this.uploadImage(
          file,
          undefined,
          (progress) => onProgress?.(i, progress)
        );

        if (result.success && result.data) {
          results.push(result.data);
        } else {
          errors.push(`文件 ${file.name}: ${result.error}`);
        }

        onComplete?.(i, result);
      } catch (error) {
        const errorMsg = `文件 ${file.name}: 上传失败`;
        errors.push(errorMsg);
        onComplete?.(i, { success: false, error: errorMsg });
      }
    }

    return {
      success: errors.length === 0,
      data: results,
      error: errors.length > 0 ? errors.join('; ') : undefined,
    };
  }

  /**
   * 删除图片
   */
  async deleteImage(itemId: string): Promise<ApiResponse<void>> {
    return await api.post<void>(API_ENDPOINTS.USER.DELETE_IMAGE, {
      item_id: itemId,
    });
  }

  /**
   * 验证文件
   */
  private validateFile(file: File): { isValid: boolean; error?: string } {
    // 检查文件类型
    const allowedTypes = [
      'image/jpeg',
      'image/jpg',
      'image/png',
      'image/webp',
      'image/gif',
    ];
    
    if (!allowedTypes.includes(file.type)) {
      return {
        isValid: false,
        error: '不支持的文件格式，请上传 JPG、PNG、WebP 或 GIF 格式的图片',
      };
    }

    // 检查文件大小 (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: '文件大小不能超过 10MB',
      };
    }

    // 检查文件名
    if (file.name.length > 255) {
      return {
        isValid: false,
        error: '文件名过长',
      };
    }

    return { isValid: true };
  }

  /**
   * 压缩图片
   */
  async compressImage(
    file: File,
    maxWidth: number = 1920,
    maxHeight: number = 1920,
    quality: number = 0.8
  ): Promise<File> {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        // 计算新尺寸
        let { width, height } = img;
        
        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height);
          width *= ratio;
          height *= ratio;
        }

        // 设置canvas尺寸
        canvas.width = width;
        canvas.height = height;

        // 绘制图片
        ctx?.drawImage(img, 0, 0, width, height);

        // 转换为Blob
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const compressedFile = new File([blob], file.name, {
                type: file.type,
                lastModified: Date.now(),
              });
              resolve(compressedFile);
            } else {
              reject(new Error('图片压缩失败'));
            }
          },
          file.type,
          quality
        );
      };

      img.onerror = () => reject(new Error('图片加载失败'));
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * 预览图片
   */
  createPreviewUrl(file: File): string {
    return URL.createObjectURL(file);
  }

  /**
   * 释放预览URL
   */
  revokePreviewUrl(url: string): void {
    URL.revokeObjectURL(url);
  }

  /**
   * 从URL创建File对象
   */
  async createFileFromUrl(url: string, filename: string): Promise<File> {
    const response = await fetch(url);
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type });
  }

  /**
   * 获取图片元数据
   */
  async getImageMetadata(file: File): Promise<{
    width: number;
    height: number;
    size: number;
    type: string;
    name: string;
  }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      
      img.onload = () => {
        resolve({
          width: img.naturalWidth,
          height: img.naturalHeight,
          size: file.size,
          type: file.type,
          name: file.name,
        });
        URL.revokeObjectURL(img.src);
      };
      
      img.onerror = () => {
        reject(new Error('无法读取图片信息'));
        URL.revokeObjectURL(img.src);
      };
      
      img.src = URL.createObjectURL(file);
    });
  }
}

// 导出单例实例
export const uploadService = new UploadService();

// 导出便捷方法
export const {
  uploadImage,
  uploadBatch,
  deleteImage,
  compressImage,
  createPreviewUrl,
  revokePreviewUrl,
  createFileFromUrl,
  getImageMetadata,
} = uploadService;