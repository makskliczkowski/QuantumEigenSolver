#include "../src/common.h"

// ########################################################				PROGRESS BAR				########################################################

void pBar::update(double newProgress)
{
	currentProgress += newProgress;
	amountOfFiller = (int)((currentProgress / neededProgress) * (double)pBarLength);
}

void pBar::print()
{
	currUpdateVal %= pBarUpdater.length();
	std::cout << "\r";															            // Bring cursor to start of line
	std::cout << firstPartOfpBar;												            // Print out first part of pBar
	for (int a = 0; a < amountOfFiller; a++) {												// Print out current progress
		std::cout << pBarFiller;                                                            // By filling the output
	}
	std::cout << pBarUpdater[currUpdateVal];
	for (int b = 0; b < pBarLength - amountOfFiller; b++) {									// Print out spaces
		std::cout << " ";
	}
	std::cout << lastPartOfpBar;												            // Print out last part of progress bar
	std::cout << " (" << (int)(100 * (currentProgress / neededProgress)) << "%)";	        // This just prints out the percent
	std::cout << std::flush;
	currUpdateVal += 1;
}

void pBar::printWithTime(std::string message)
{
	{
		std::cout << "\t\t\t\t-> time: " << t_s(timer) << message << " : \n";
		this->print();
		std::cout << std::endl;
	}
	this->update(percentage);
}

// ########################################################				PROGRESS BAR				########################################################
