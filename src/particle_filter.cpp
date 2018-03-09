/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 20;
	
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i< num_particles; i++)
	{
		Particle particle;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		
		particles.push_back(particle);
		weights.push_back(1.0);
	}
	
	is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	
	for (Particle& particle : particles)
	{
		if (fabs(yaw_rate) < 0.001)
		{
			particle.x += velocity*delta_t*cos(particle.theta);
			particle.y += velocity*delta_t*sin(particle.theta);
		}
		else
		{
			particle.x += velocity/yaw_rate * (sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
			particle.y += velocity/yaw_rate * (cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
			particle.theta += yaw_rate*delta_t;
			
		}
		
		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
		
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

	}

}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (LandmarkObs& observation : observations)
	{
		double min_distance = std::numeric_limits<double>::max();
		for (LandmarkObs prediction : predicted)
		{
			double distance = dist(observation.x,observation.y,prediction.x,prediction.y);
			if (distance< min_distance)
			{
				min_distance = distance;
				observation.id = prediction.id;
			}
			
		}
	}

}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	
	
	for (int i = 0; i< particles.size(); i++)
	{
		Particle particle = particles[i];
		double probability = 1.0;
		std::vector<LandmarkObs> map_coordinate_observations = ToMapCoordinates(particle, observations);
		std::vector<LandmarkObs> nearby_landmarks = GetNearbyLandmarks(particle, map_landmarks, sensor_range);
		dataAssociation(nearby_landmarks,map_coordinate_observations );
		for (LandmarkObs observation: map_coordinate_observations)
		{
			//std::cout << "observation.id : " << observation.id << std::endl;
			for(LandmarkObs landmark : nearby_landmarks)
			{
				if (observation.id == landmark.id)
				{
					probability *= GetProbability(observation, landmark, std_landmark);
				}
			}
		}
		weights[i] = probability;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	std::vector<double> normalized;

	double max = *std::max_element(weights.begin(), weights.end());
	
	for (double weight : weights)
	{
		normalized.push_back(weight*100/max);
	}
	
	std::vector<Particle> resampled;
	uniform_real_distribution<double> beta_dist(0.0, 100);
	while(resampled.size() < particles.size())
	{
		double beta = beta_dist(gen);
		int index = 0;
		while (beta > normalized[index])
		{
			beta -= normalized[index];
			index = (index+1)%normalized.size();
		}
		resampled.push_back(particles[index]);
	}
	particles = resampled;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	// Not used
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::vector<LandmarkObs> ParticleFilter::ToMapCoordinates(Particle particle, std::vector<LandmarkObs> observations)
{
	std::vector<LandmarkObs> map_coordinate_observations;
	for (LandmarkObs observation : observations)
	{
		LandmarkObs map_coordinate_cbservation = observation;
		map_coordinate_cbservation.x = particle.x + cos(particle.theta)*observation.x - sin(particle.theta)* observation.y;
		map_coordinate_cbservation.y = particle.y + sin(particle.theta)*observation.x + cos(particle.theta)* observation.y;
		map_coordinate_cbservation.id = observation.id;

		map_coordinate_observations.push_back(map_coordinate_cbservation);
	}
	return map_coordinate_observations;
}

std::vector<LandmarkObs> ParticleFilter::GetNearbyLandmarks(Particle particle
															,const Map &landmarks, double sensor_range)
{
	std::vector<LandmarkObs> nearby_landmarks;
	for (Map::single_landmark_s landmark : landmarks.landmark_list)
	{
		if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f)<sensor_range)
		{
			LandmarkObs nearby_landmark;
			nearby_landmark.x = landmark.x_f;
			nearby_landmark.y = landmark.y_f;
			nearby_landmark.id = landmark.id_i;
			nearby_landmarks.push_back(nearby_landmark);
		}
	}
	return nearby_landmarks;
}

double ParticleFilter::GetProbability(LandmarkObs observation, LandmarkObs landmark, double std_landmark[])
{
	
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double delta_x = observation.x - landmark.x;
	double delta_y = observation.y - landmark.y;
	double prob = 1.0/(2*M_PI*std_x*std_y) * exp(-(delta_x*delta_x/(2*std_x*std_x)+delta_y*delta_y/(2*std_y*std_y)));

	return prob;
}
