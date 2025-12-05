import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';

// Import Screens
import SplashScreen from './screens/SplashScreen';
import LoginScreen from './screens/LoginScreen';
import MainMenuScreen from './screens/MainMenuScreen';
import CameraScreen from './screens/CameraScreen';
import ResultsScreen from './screens/ResultsScreen';
import ReportScreen from './screens/ReportScreen';
import DiameterSelectionScreen from './screens/DiameterSelectionScreen';
import ImageSourceScreen from './screens/ImageSourceScreen';
import HistoryScreen from './screens/HistoryScreen';
import TestDetailsScreen from './screens/TestDetailsScreen';
import RibTestNavigator from './src/screens/RibTestNavigator';
import SettingsScreen from './screens/SettingsScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator
        initialRouteName="Splash"
        screenOptions={{
          headerShown: false,
          cardStyle: { backgroundColor: '#fff' },
          cardOverlayEnabled: false,
          gestureEnabled: false,
          animationEnabled: true,
          presentation: 'card',
        }}
      >
        <Stack.Screen name="Splash" component={SplashScreen} />
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="MainMenu" component={MainMenuScreen} />
        <Stack.Screen name="DiameterSelection" component={DiameterSelectionScreen} />
        <Stack.Screen name="ImageSource" component={ImageSourceScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="Results" component={ResultsScreen} />
        <Stack.Screen name="Report" component={ReportScreen} />
        <Stack.Screen name="History" component={HistoryScreen} />
        <Stack.Screen name="TestDetails" component={TestDetailsScreen} />
        <Stack.Screen 
          name="RibTest" 
          component={RibTestNavigator}
          options={{
            cardStyle: { backgroundColor: '#0A0A0A' },
          }}
        />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}