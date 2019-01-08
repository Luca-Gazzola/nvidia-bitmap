// System includes
#include <iostream>
// File includes
#include "../include/BitmapException.h"



BitmapException::BitmapException()
    : std::runtime_error("[ERROR] Unknown illegal action has occurred")
{}

BitmapException::BitmapException(const std::string& message)
    : std::runtime_error(message)
{}