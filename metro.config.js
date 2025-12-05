const { getDefaultConfig } = require('@expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver = config.resolver || {};
config.resolver.alias = Object.assign({}, config.resolver.alias, {
  '@react-native/codegen/lib/parsers/flow/parser': require.resolve('hermes-parser'),
  '@react-native/codegen/lib/parsers/flow': require.resolve('hermes-parser'),
});

module.exports = config;


