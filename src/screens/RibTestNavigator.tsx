import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import UserGuidelineScreen from './UserGuidelineScreen';
import HomeScreen from './HomeScreen';
import CameraScreen from './CameraScreen';
import { ARValueSimpleScreen, ARValueDetailedScreen } from './ARValueScreen';

export type RibTestStackParamList = {
  UserGuidelines: undefined;
  Home: undefined;
  Camera: undefined;
  ARValue: { results: any };
  ARValueDetailed: { angleResults: any, heightResults: any, diameter: any, rows: any };
};

const Stack = createStackNavigator<RibTestStackParamList>();

const RibTestNavigator: React.FC = () => {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: false,
          cardStyle: { backgroundColor: '#0A0A0A' },
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="UserGuidelines" component={UserGuidelineScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="ARValue" component={ARValueSimpleScreen} />
        <Stack.Screen name="ARValueDetailed" component={ARValueDetailedScreen} />
      </Stack.Navigator>
    </SafeAreaProvider>
  );
};

export default RibTestNavigator; 