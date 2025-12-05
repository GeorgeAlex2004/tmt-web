import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import LoadingAnimation from './LoadingAnimation';

const LoadingTest = () => {
  const [isLoading, setIsLoading] = useState(false);

  const startLoading = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
    }, 30000);
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#0D47A1', '#1976D2']}
        style={styles.header}
      >
        <Text style={styles.title}>Loading Animation Test</Text>
        <Text style={styles.subtitle}>Test the beautiful loading animation</Text>
      </LinearGradient>

      <View style={styles.content}>
        <TouchableOpacity
          style={styles.button}
          onPress={startLoading}
          disabled={isLoading}
        >
          <LinearGradient
            colors={['#4CAF50', '#388E3C']}
            style={styles.buttonGradient}
          >
            <Text style={styles.buttonText}>
              {isLoading ? 'Processing...' : 'Start Loading Animation'}
            </Text>
          </LinearGradient>
        </TouchableOpacity>

        <Text style={styles.info}>
          This will show the loading animation for 30 seconds to demonstrate all the features:
        </Text>
        
        <View style={styles.featureList}>
          <Text style={styles.featureItem}>• Animated progress bar</Text>
          <Text style={styles.featureItem}>• Step-by-step progress indicators</Text>
          <Text style={styles.featureItem}>• Pulsing camera icon</Text>
          <Text style={styles.featureItem}>• Rotating animations</Text>
          <Text style={styles.featureItem}>• Smooth fade and scale effects</Text>
        </View>
      </View>

      <LoadingAnimation 
        visible={isLoading}
        title="Processing Image..."
        subtitle="SAM model processing may take 2-3 minutes"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    paddingVertical: 40,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  content: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  button: {
    marginBottom: 30,
    borderRadius: 15,
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
  },
  buttonGradient: {
    padding: 20,
    alignItems: 'center',
  },
  buttonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  info: {
    fontSize: 16,
    color: '#333',
    marginBottom: 20,
    textAlign: 'center',
  },
  featureList: {
    backgroundColor: '#E3F2FD',
    padding: 20,
    borderRadius: 15,
  },
  featureItem: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
});

export default LoadingTest; 