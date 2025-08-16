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
    checkExistingAuth();
  }, []);

  const checkExistingAuth = async () => {
    try {
      setIsLoading(true);
      
      // Check multiple storage locations for token
      const storageToken = localStorage.getItem('token') || 
                          sessionStorage.getItem('token');
      
      // Try to extract token from cookies as fallback
      const cookieToken = getCookieValue('access_token') || getCookieValue('token');
      
      const foundToken = storageToken || cookieToken;
      
      if (foundToken) {
        const isValid = await validateToken(foundToken);
        if (isValid) {
          setToken(foundToken);
          setIsAuthenticated(true);
          // Store token in preferred location
          localStorage.setItem('token', foundToken);
        } else {
          clearAuthData();
        }
      }
    } catch (error) {
      clearAuthData();
    } finally {
      setIsLoading(false);
    }
  };

  const validateToken = async (token) => {
    try {
      // Create a timeout promise
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Token validation timeout')), 10000);
      });
      
      // Create the fetch promise - no auth needed since endpoint is now public
      const fetchPromise = fetch('/api/current-status', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      // Race between fetch and timeout
      const response = await Promise.race([fetchPromise, timeoutPromise]);
      
      if (response.ok) {
        return true;
      } else {
        return false;
      }
    } catch (error) {
      if (error.message === 'Token validation timeout') {
        // Timeout error
      }
      return false;
    }
  };

  const login = async (username, password) => {
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.access_token) {
          // Store the token in localStorage for persistence
          localStorage.setItem('token', data.access_token);
          
          // Update auth state
          setToken(data.access_token);
          setIsAuthenticated(true);
          setUser(username);
          
          return { success: true };
        } else {
          return { success: false, error: 'No token received' };
        }
      } else {
        const errorData = await response.json();
        return { success: false, error: errorData.message || 'Login failed' };
      }
    } catch (error) {
      return { success: false, error: 'Network error. Please try again.' };
    }
  };

  const logout = () => {
    clearAuthData();
    
    // Also try to clear any server-side session
    fetch('/api/logout', {
      method: 'POST',
      credentials: 'include',
    }).catch(() => {
      // Ignore errors - we're logging out anyway
    });
  };

  const clearAuthData = () => {
    // Clear all storage locations
    localStorage.removeItem('token');
    sessionStorage.removeItem('token');
    
    // Clear cookies by setting them to expire
    document.cookie = 'access_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    
    // Clear state
    setToken(null);
    setIsAuthenticated(false);
    setUser(null);
  };

  // Utility function to get cookie value
  const getCookieValue = (name) => {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
  };

  const value = {
    isAuthenticated,
    isLoading,
    user,
    token,
    login,
    logout,
    validateToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
