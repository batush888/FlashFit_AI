import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import Upload from '../Upload/Upload';
import { useNotificationStore } from '../../stores/notificationStore';
import { useAuthStore } from '../../stores/authStore';

// Mock the stores
jest.mock('../../stores/notificationStore');
jest.mock('../../stores/authStore');
jest.mock('../../services/api', () => ({
  uploadClothing: jest.fn(),
}));

const mockAddNotification = jest.fn();
const mockUser = {
  id: 'user-123',
  email: 'test@example.com',
  full_name: 'Test User'
};

(useNotificationStore as jest.Mock).mockReturnValue({
  addNotification: mockAddNotification
});

(useAuthStore as jest.Mock).mockReturnValue({
  user: mockUser,
  token: 'mock-token'
});

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Upload Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders upload form correctly', () => {
    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    expect(screen.getByText(/upload clothing item/i)).toBeInTheDocument();
    expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /browse files/i })).toBeInTheDocument();
  });

  it('handles file selection', async () => {
    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText('test.jpg')).toBeInTheDocument();
    });
  });

  it('validates file type', async () => {
    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(mockAddNotification).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          title: 'Invalid File Type'
        })
      );
    });
  });

  it('validates file size', async () => {
    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    // Create a large file (mock)
    const largeFile = new File(['x'.repeat(11 * 1024 * 1024)], 'large.jpg', { type: 'image/jpeg' });
    Object.defineProperty(largeFile, 'size', { value: 11 * 1024 * 1024 });

    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    await waitFor(() => {
      expect(mockAddNotification).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          title: 'File Too Large'
        })
      );
    });
  });

  it('handles drag and drop', async () => {
    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const dropZone = screen.getByText(/drag and drop/i).closest('div');
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    const dropEvent = new Event('drop', { bubbles: true });
    Object.defineProperty(dropEvent, 'dataTransfer', {
      value: {
        files: [file]
      }
    });

    fireEvent(dropZone!, dropEvent);

    await waitFor(() => {
      expect(screen.getByText('test.jpg')).toBeInTheDocument();
    });
  });

  it('submits form with valid data', async () => {
    const mockUploadClothing = require('../../services/api').uploadClothing;
    mockUploadClothing.mockResolvedValue({
      id: 'clothing-123',
      category: 'shirt',
      color: 'blue'
    });

    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    // Fill in metadata
    const categorySelect = screen.getByLabelText(/category/i);
    fireEvent.change(categorySelect, { target: { value: 'shirt' } });

    const colorInput = screen.getByLabelText(/color/i);
    fireEvent.change(colorInput, { target: { value: 'blue' } });

    const submitButton = screen.getByRole('button', { name: /upload/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockUploadClothing).toHaveBeenCalledWith(
        expect.any(FormData),
        'mock-token'
      );
    });

    await waitFor(() => {
      expect(mockAddNotification).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'success',
          title: 'Upload Successful'
        })
      );
    });
  });

  it('handles upload error', async () => {
    const mockUploadClothing = require('../../services/api').uploadClothing;
    mockUploadClothing.mockRejectedValue(new Error('Upload failed'));

    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    const submitButton = screen.getByRole('button', { name: /upload/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockAddNotification).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          title: 'Upload Failed'
        })
      );
    });
  });

  it('shows loading state during upload', async () => {
    const mockUploadClothing = require('../../services/api').uploadClothing;
    mockUploadClothing.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 1000)));

    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    const submitButton = screen.getByRole('button', { name: /upload/i });
    fireEvent.click(submitButton);

    expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    expect(submitButton).toBeDisabled();
  });

  it('clears form after successful upload', async () => {
    const mockUploadClothing = require('../../services/api').uploadClothing;
    mockUploadClothing.mockResolvedValue({
      id: 'clothing-123',
      category: 'shirt',
      color: 'blue'
    });

    render(
      <TestWrapper>
        <Upload />
      </TestWrapper>
    );

    const fileInput = screen.getByLabelText(/choose file/i) as HTMLInputElement;
    const file = new File(['test image'], 'test.jpg', { type: 'image/jpeg' });

    fireEvent.change(fileInput, { target: { files: [file] } });

    const submitButton = screen.getByRole('button', { name: /upload/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockUploadClothing).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(fileInput.files).toHaveLength(0);
    });
  });
});