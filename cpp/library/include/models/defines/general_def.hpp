/**
*   file: cpp/library/include/models/defines/general_def.hpp
*   brief: General definitions for the models to obtain the properties of the Hamiltonian from the simulations;
*   version: 1.0
*/


#include <vector>
#include <map>
#include <cmath>

namespace General_types
{
    /**
    * @brief Default model utility structure providing common lookup and getter functions.
    * 
    * This template struct is intended to be inherited by model classes, supplying
    * static utility methods for parameter lookup and value retrieval from maps of
    * the form `std::map<double, std::vector<double>>`. It provides:
    * 
    * - Index calculation for a given system size (`Ns`) within a specified range.
    * - Lookup of the closest value in a map for a given parameter, with tie-breaking.
    * - Convenience getters for bandwidth, variance, mean level spacing, and Thouless energy,
    *   which delegate to the closest-value lookup using static members of the derived class.
    * 
    * The derived class must define the following static members:
    *   - `static const std::map<double, std::vector<double>> bandwidth;`
    *   - `static const std::map<double, std::vector<double>> variance;`
    *   - `static const std::map<double, std::vector<double>> mean_lvl_spacing;`
    *   - `static const std::map<double, std::vector<double>> thouless;`
    * 
    * @tparam Derived The model class inheriting from this struct.
    */
    template <class Derived>
    struct Model_default
    {
        // compute vector‐index from N
        static int getIndex (int _Ns, int _minNs = 7, int _maxNs = 16)
        {
            return std::clamp(_Ns - _minNs, 0, _maxNs - _minNs);
        }

        // nearest‐α lookup in any map<double,vector<double>>
        template <typename MapT>
        static double getClosestValue(  const MapT& _map,
                                        double   _param,
                                        int      _ns,
                                        int      _minNs = 7,
                                        int      _maxNs = 16)
        {
            // find the index for Ns
            int idx             = getIndex(_ns, _minNs, _maxNs);
            if (idx < 0)        return NAN;
            if (_map.empty())   return NAN;

            // all keys < alpha → last key
            auto it_hi      = _map.lower_bound(_param);
            if (it_hi == _map.end())
            {
                const auto it = std::prev(_map.end());
                return it->second[idx];
            }
            
            // all keys ≥ alpha → first key
            if (it_hi == _map.begin())              
            {
                return it_hi->second[idx];
            }
            
            // otherwise compare distances to the two neighbors
            auto it_lo  = std::prev(it_hi);
            double d_lo = _param - it_lo->first;
            double d_hi = it_hi->first - _param;

            // tie‐break to the lower key
            return (d_lo <= d_hi
                    ? it_lo->second[idx]
                    : it_hi->second[idx]);
        }

        // now each getter just calls getClosestValue
        static double getBandwidth      (double a, int Ns) { return getClosestValue(Derived::bandwidth_,         a, Ns, Derived::minNs_, Derived::maxNs_); };
        static double getVariance       (double a, int Ns) { return getClosestValue(Derived::variance_,          a, Ns, Derived::minNs_, Derived::maxNs_); };
        static double getMeanLvlSpacing (double a, int Ns) { return getClosestValue(Derived::mean_lvl_spacing,  a, Ns, Derived::minNs_, Derived::maxNs_); };
        static double getThouless       (double a, int Ns) { return getClosestValue(Derived::thouless,          a, Ns, Derived::minNs_, Derived::maxNs_); };
    };
}

// ----------------------------------------------------
