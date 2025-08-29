import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('http://localhost:3000');
  });

  test('should display login form', async ({ page }) => {
    // Check if login form elements are present
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should show validation errors for empty form', async ({ page }) => {
    // Try to submit empty form
    await page.click('button[type="submit"]');
    
    // Check for validation messages
    await expect(page.locator('text=Email is required')).toBeVisible();
    await expect(page.locator('text=Password is required')).toBeVisible();
  });

  test('should show error for invalid email format', async ({ page }) => {
    // Fill invalid email
    await page.fill('input[type="email"]', 'invalid-email');
    await page.fill('input[type="password"]', 'password123');
    await page.click('button[type="submit"]');
    
    // Check for email validation error
    await expect(page.locator('text=Please enter a valid email')).toBeVisible();
  });

  test('should navigate to register page', async ({ page }) => {
    // Click register link
    await page.click('text=Don\'t have an account? Sign up');
    
    // Check if we're on register page
    await expect(page.locator('text=Create Account')).toBeVisible();
    await expect(page.locator('input[name="fullName"]')).toBeVisible();
  });

  test('should register new user successfully', async ({ page }) => {
    // Navigate to register page
    await page.click('text=Don\'t have an account? Sign up');
    
    // Fill registration form
    await page.fill('input[name="fullName"]', 'Test User');
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'securepassword123');
    await page.fill('input[name="confirmPassword"]', 'securepassword123');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Check for success message or redirect
    await expect(page.locator('text=Registration successful')).toBeVisible({ timeout: 10000 });
  });

  test('should login with valid credentials', async ({ page }) => {
    // Fill login form with valid credentials
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'securepassword123');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Check if redirected to dashboard
    await expect(page.locator('text=Dashboard')).toBeVisible({ timeout: 10000 });
    await expect(page.url()).toContain('/dashboard');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    // Fill login form with invalid credentials
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'wrongpassword');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Check for error message
    await expect(page.locator('text=Invalid email or password')).toBeVisible();
  });

  test('should logout successfully', async ({ page }) => {
    // First login
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'securepassword123');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard
    await expect(page.locator('text=Dashboard')).toBeVisible({ timeout: 10000 });
    
    // Click logout
    await page.click('button:has-text("Logout")');
    
    // Check if redirected to login
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.url()).toContain('/login');
  });

  test('should persist login state on page refresh', async ({ page }) => {
    // Login first
    await page.fill('input[type="email"]', 'test@example.com');
    await page.fill('input[type="password"]', 'securepassword123');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard
    await expect(page.locator('text=Dashboard')).toBeVisible({ timeout: 10000 });
    
    // Refresh page
    await page.reload();
    
    // Should still be logged in
    await expect(page.locator('text=Dashboard')).toBeVisible();
  });

  test('should redirect to login when accessing protected route', async ({ page }) => {
    // Try to access protected route directly
    await page.goto('http://localhost:3000/wardrobe');
    
    // Should be redirected to login
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.url()).toContain('/login');
  });
});