import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  // Check for existing authentication on mount
  useEffect(() => {
    console.log('🔐 AuthProvider: Checking for existing authentication...');
    checkExistingAuth();
  }, []);

  const checkExistingAuth = async () => {
    try {
      setIsLoading(true);
      
      // Check multiple storage locations for token
      const storageToken = localStorage.getItem('token') || 
                          sessionStorage.getItem('token') || 
                          localStorage.getItem('access_token') ||
                          sessionStorage.getItem('access_token');
      
      // Also check for cookies
      const cookieToken = getCookieValue('access_token') || getCookieValue('token');
      
      const foundToken = storageToken || cookieToken;
      
      console.log('🔐 Found token in storage:', !!storageToken);
      console.log('🔐 Found token in cookies:', !!cookieToken);
      console.log('🔐 Using token:', !!foundToken);
      
      if (foundToken) {
        const isValid = await validateToken(foundToken);
        if (isValid) {
          console.log('🔐 Token is valid, setting authenticated state');
          setToken(foundToken);
          setIsAuthenticated(true);
          // Store token in preferred location
          localStorage.setItem('token', foundToken);
        } else {
          console.log('🔐 Token is invalid, clearing auth');
          clearAuthData();
        }
      } else {
        console.log('🔐 No token found');
      }
    } catch (error) {
      console.error('🔐 Error checking existing auth:', error);
      clearAuthData();
    } finally {
      setIsLoading(false);
    }
  };

  const validateToken = async (tokenToValidate) => {
    try {
      console.log('🔐 Validating token with backend...');
      
      // Enhanced token validation with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch('http://localhost:5000/api/data?limit=1', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${tokenToValidate}`,
          'Content-Type': 'application/json',
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        console.log('🔐 Token validation successful');
        return true;
      } else {
        console.log('🔐 Token validation failed:', response.status);
        return false;
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('🔐 Token validation timed out');
      } else {
        console.error('🔐 Token validation error:', error);
      }
      return false;
    }
  };

  const login = async (credentials) => {
    try {
      console.log('🔐 AuthContext: Attempting login...');
      setIsLoading(true);
      
      const response = await fetch('http://localhost:5000/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
        credentials: 'include', // Include cookies in request
      });

      if (response.ok) {
        const data = await response.json();
        console.log('🔐 Login successful, storing auth data');
        
        const accessToken = data.access_token;
        
        // Store token in multiple locations for reliability
        localStorage.setItem('token', accessToken);
        localStorage.setItem('access_token', accessToken);
        sessionStorage.setItem('token', accessToken);
        
        // Also set as cookie manually if backend doesn't
        setCookie('access_token', accessToken, 7); // 7 days
        setCookie('token', accessToken, 7);
        
        setToken(accessToken);
        setIsAuthenticated(true);
        setUser({ username: credentials.username });
        
        console.log('🔐 Auth state updated successfully');
        return { success: true };
      } else {
        const errorData = await response.json();
        console.error('🔐 Login failed:', errorData);
        return { success: false, error: errorData.message || 'Login failed' };
      }
    } catch (error) {
      console.error('🔐 Login error:', error);
      return { success: false, error: 'Network error occurred' };
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    console.log('🔐 AuthContext: Logging out...');
    clearAuthData();
  };

  const clearAuthData = () => {
    // Clear all possible storage locations
    localStorage.removeItem('token');
    localStorage.removeItem('access_token');
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('access_token');
    
    // Clear cookies
    deleteCookie('access_token');
    deleteCookie('token');
    
    setToken(null);
    setIsAuthenticated(false);
    setUser(null);
    
    console.log('🔐 All auth data cleared');
  };

  // Helper function to get cookie value
  const getCookieValue = (name) => {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
  };

  // Helper function to set cookie
  const setCookie = (name, value, days) => {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
  };

  // Helper function to delete cookie
  const deleteCookie = (name) => {
    document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
  };

  const value = {
    isAuthenticated,
    isLoading,
    user,
    token,
    login,
    logout,
    checkExistingAuth
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
